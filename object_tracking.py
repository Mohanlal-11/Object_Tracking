import argparse
import os
import sys
import time

import cv2
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

logger = logging.getLogger()

from tracker import KalmanFilter
from LightDetector.util import cfg, load_config
from tools.model_inference import Predictor


class MultiObjectTracker:
    def __init__(self, distance_threshold=60.0, max_age=10, min_hits=1):
        self.distance_threshold = distance_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 1

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2, _ = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return np.array([cx, cy], dtype=np.float32)

    def _predicted_center(self, track):
        prediction = track["kf"].predict()
        return np.array([prediction[0, 0], prediction[1, 0]], dtype=np.float32)

    def _predict_box(self, track):
        pred = track["kf"].predict()
        cx = float(pred[0, 0])
        cy = float(pred[1, 0])
        w = max(track["bbox"][2] - track["bbox"][0], 10.0)
        h = max(track["bbox"][3] - track["bbox"][1], 10.0)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def update(self, detections):
        detections = [np.asarray(det, dtype=np.float32).copy() for det in detections]

        if not detections:
            for track in self.tracks:
                track["kf"].predict()
                track["age"] += 1
                track["time_since_update"] += 1
            self.tracks = [t for t in self.tracks if t["time_since_update"] <= self.max_age]
            return []

        if not self.tracks:
            matched = []
            unmatched_dets = list(range(len(detections)))
        else:
            predicted_centers = [self._predicted_center(t) for t in self.tracks]
            cost_matrix = []
            for ti, track in enumerate(self.tracks):
                for di, det in enumerate(detections):
                    det_center = self._center(det)
                    cost = np.linalg.norm(predicted_centers[ti] - det_center)
                    cost_matrix.append((cost, ti, di))

            cost_matrix.sort(key=lambda x: x[0])
            matched = []
            matched_tracks = set()
            matched_dets = set()
            for cost, ti, di in cost_matrix:
                if ti in matched_tracks or di in matched_dets:
                    continue
                if cost <= self.distance_threshold:
                    matched.append((ti, di))
                    matched_tracks.add(ti)
                    matched_dets.add(di)

            unmatched_tracks = [ti for ti in range(len(self.tracks)) if ti not in matched_tracks]
            unmatched_dets = [di for di in range(len(detections)) if di not in matched_dets]

            for ti in unmatched_tracks:
                self.tracks[ti]["age"] += 1
                self.tracks[ti]["time_since_update"] += 1
                self.tracks[ti]["bbox"] = self._predict_box(self.tracks[ti])

        for ti, di in matched:
            track = self.tracks[ti]
            det = detections[di]
            center = self._center(det)
            track["kf"].update(center)
            track["bbox"] = det.copy()
            track["hits"] += 1
            track["age"] = 0
            track["time_since_update"] = 0

        for di in unmatched_dets:
            det = detections[di]
            center = self._center(det)
            kf = KalmanFilter(dt=1.0)
            kf.update(center)
            self.tracks.append({
                "id": self.next_id,
                "bbox": det.copy(),
                "kf": kf,
                "hits": 1,
                "age": 0,
                "time_since_update": 0,
            })
            self.next_id += 1

        alive_tracks = []
        for track in self.tracks:
            if track["time_since_update"] <= self.max_age and track["hits"] >= self.min_hits:
                alive_tracks.append(track)

        self.tracks = alive_tracks
        return self.tracks


def get_detection_boxes(results):
    detections = []
    if not results:
        return detections

    result_for_frame = results.get(0, {})
    for cls_id, boxes in result_for_frame.items():
        for box in boxes:
            if len(box) < 5:
                continue
            x1, y1, x2, y2, score = map(float, box[:5])
            detections.append([x1, y1, x2, y2, score])
    return detections


def draw_tracks(frame, tracks, color=(0, 255, 0)):
    out = frame.copy()
    for track in tracks:
        # print(track["bbox"])
        x1, y1, x2, y2 = np.asarray(track["bbox"][:4], dtype=np.int32)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"ID {track['id']}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Run object detection model with Kalman-filter multi-object tracking")
    parser.add_argument("--cfg", type=str, default="config/configuration.yml", help="Path to config YAML")
    parser.add_argument("--model", type=str, default="workspace/weights/model_best/best_weight.pth", help="Model weight path")
    parser.add_argument("--vid_path", type=str, default="video2.mp4", help="Input video path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference")
    parser.add_argument("--thres", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--gray", action="store_true", help="Run on grayscale inputs")
    parser.add_argument("--camera", action="store_true", help="Use webcam instead of a file")
    parser.add_argument("--save", action="store_true", help="Save tracked video")
    parser.add_argument("--out_name", type=str, default="tracked_output.mp4", help="Saved output video path")
    args = parser.parse_args()

    load_config(cfg, args.cfg)
    predictor = Predictor(cfg, args.model, logger=logger, device=args.device)
    tracker = MultiObjectTracker(distance_threshold=40.0, max_age=10, min_hits=1)

    if args.camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.vid_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open video source.")

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.out_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        meta, results = predictor.inference(frame, args.thres, gray=args.gray)
        detections = get_detection_boxes(results)
        tracked = tracker.update(detections)
        output = draw_tracks(frame, tracked)

        fps_text = f"FPS: {1.0 / max(time.perf_counter() - start, 1e-6):.1f}"
        cv2.putText(output, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if writer is not None:
            writer.write(output)
        # output = cv2.resize(output, (640,640))
        cv2.imshow("Object Tracking", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
