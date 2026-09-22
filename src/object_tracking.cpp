#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
// #include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace {

constexpr int kInputWidth = 320;
constexpr int kInputHeight = 320;
constexpr int kNumClasses = 1;
constexpr int kRegMax = 7;
constexpr int kOutputChannels = kNumClasses + 4 * (kRegMax + 1);
constexpr float kNmsThreshold = 0.3F;

struct Detection {
    cv::Rect2f box;
    float score = 0.0F;
};

struct KalmanFilter {
    std::array<float, 4> state{0.0F, 0.0F, 0.0F, 0.0F};
    std::array<float, 16> covariance{};

    KalmanFilter(float x, float y) {
        covariance.fill(0.0F);
        for (int i = 0; i < 4; ++i) {
            covariance[i * 4 + i] = 1000.0F;
        }
        state[0] = x;
        state[1] = y;
    }

    std::array<float, 2> predict() {
        state[0] += state[2];
        state[1] += state[3];

        const float transition[4][4] = {
            {1.0F, 0.0F, 1.0F, 0.0F},
            {0.0F, 1.0F, 0.0F, 1.0F},
            {0.0F, 0.0F, 1.0F, 0.0F},
            {0.0F, 0.0F, 0.0F, 1.0F},
        };
        std::array<float, 16> predicted{};
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                for (int first = 0; first < 4; ++first) {
                    for (int second = 0; second < 4; ++second) {
                        predicted[row * 4 + column] += transition[row][first]
                            * covariance[first * 4 + second] * transition[column][second];
                    }
                }
            }
        }
        covariance = predicted;
        for (int i = 0; i < 4; ++i) {
            covariance[i * 4 + i] += 0.1F;
        }
        return {state[0], state[1]};
    }

    void update(float measured_x, float measured_y) {
        const float s00 = covariance[0] + 10.0F;
        const float s01 = covariance[1];
        const float s10 = covariance[4];
        const float s11 = covariance[5] + 10.0F;
        const float determinant = s00 * s11 - s01 * s10;
        if (std::fabs(determinant) < 1e-6F) {
            return;
        }

        const float inv00 = s11 / determinant;
        const float inv01 = -s01 / determinant;
        const float inv10 = -s10 / determinant;
        const float inv11 = s00 / determinant;
        const float innovation_x = measured_x - state[0];
        const float innovation_y = measured_y - state[1];

        std::array<float, 8> gain{};
        for (int row = 0; row < 4; ++row) {
            const float p0 = covariance[row * 4];
            const float p1 = covariance[row * 4 + 1];
            gain[row * 2] = p0 * inv00 + p1 * inv10;
            gain[row * 2 + 1] = p0 * inv01 + p1 * inv11;
            state[row] += gain[row * 2] * innovation_x + gain[row * 2 + 1] * innovation_y;
        }

        const auto old_covariance = covariance;
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                covariance[row * 4 + column] = old_covariance[row * 4 + column]
                    - gain[row * 2] * old_covariance[column]
                    - gain[row * 2 + 1] * old_covariance[4 + column];
            }
        }
    }
};

struct Track {
    int id = 0;
    Detection detection;
    KalmanFilter filter;
    int hits = 1;
    int time_since_update = 0;

        Track(int track_id, const Detection& initial)
        : id(track_id),
          detection(initial),
          filter(initial.box.x + initial.box.width / 2.0F,
                                 initial.box.y + initial.box.height / 2.0F) {
                filter.update(initial.box.x + initial.box.width / 2.0F,
                                            initial.box.y + initial.box.height / 2.0F);
        }
};

float center_distance(const std::array<float, 2>& predicted_center, const Detection& detection) {
    const float detection_x = detection.box.x + detection.box.width / 2.0F;
    const float detection_y = detection.box.y + detection.box.height / 2.0F;
    const float dx = predicted_center[0] - detection_x;
    const float dy = predicted_center[1] - detection_y;
    return std::sqrt(dx * dx + dy * dy);
}

class MultiObjectTracker {
public:
    MultiObjectTracker(float distance_threshold, int max_age, int min_hits)
        : distance_threshold_(distance_threshold), max_age_(max_age), min_hits_(min_hits) {}

    const std::vector<Track>& update(const std::vector<Detection>& detections) {
        if (detections.empty()) {
            for (auto& track : tracks_) {
                track.filter.predict();
                ++track.time_since_update;
            }
            remove_dead_tracks();
            return tracks_;
        }

        std::vector<std::tuple<float, std::size_t, std::size_t>> costs;
        std::vector<std::array<float, 2>> predicted_centers;
        predicted_centers.reserve(tracks_.size());
        for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
            predicted_centers.push_back(tracks_[track_index].filter.predict());
            for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
                costs.emplace_back(center_distance(predicted_centers[track_index], detections[detection_index]),
                                   track_index, detection_index);
            }
        }
        std::sort(costs.begin(), costs.end());

        std::vector<bool> matched_tracks(tracks_.size(), false);
        std::vector<bool> matched_detections(detections.size(), false);
        for (const auto& [cost, track_index, detection_index] : costs) {
            if (cost > distance_threshold_ || matched_tracks[track_index] || matched_detections[detection_index]) {
                continue;
            }
            matched_tracks[track_index] = true;
            matched_detections[detection_index] = true;
            auto& track = tracks_[track_index];
            const auto& detection = detections[detection_index];
            track.filter.update(detection.box.x + detection.box.width / 2.0F,
                                detection.box.y + detection.box.height / 2.0F);
            track.detection = detection;
            ++track.hits;
            track.time_since_update = 0;
        }

        for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
            if (!matched_tracks[track_index]) {
                auto& track = tracks_[track_index];
                track.detection.box.x = predicted_centers[track_index][0] - track.detection.box.width / 2.0F;
                track.detection.box.y = predicted_centers[track_index][1] - track.detection.box.height / 2.0F;
                ++track.time_since_update;
            }
        }

        for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
            if (!matched_detections[detection_index]) {
                tracks_.emplace_back(next_id_++, detections[detection_index]);
            }
        }
        remove_dead_tracks();
        return tracks_;
    }

private:
    void remove_dead_tracks() {
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [this](const Track& track) {
            return track.time_since_update > max_age_ || track.hits < min_hits_;
        }), tracks_.end());
    }

    float distance_threshold_;
    int max_age_;
    int min_hits_;
    int next_id_ = 1;
    std::vector<Track> tracks_;
};

float box_iou(const cv::Rect2f& first, const cv::Rect2f& second) {
    const float intersection = (first & second).area();
    const float union_area = first.area() + second.area() - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

std::vector<Detection> decode_output(const std::vector<float>& output, const cv::Size& original_size,
                                     float score_threshold) {
    if (output.size() != 2100U * kOutputChannels) {
        throw std::runtime_error("Unexpected ONNX output size; expected 1 x 2100 x 33.");
    }
    const auto output_value = [&output](int location, int channel) {
        return output[static_cast<std::size_t>(location * kOutputChannels + channel)];
    };
    const std::array<int, 3> strides{8, 16, 32};
    std::vector<Detection> candidates;
    for (int stride : strides) {
        const int grid_width = kInputWidth / stride;
        const int grid_height = kInputHeight / stride;
        const int level_offset = stride == 8 ? 0 : (stride == 16 ? 1600 : 2000);
        for (int y = 0; y < grid_height; ++y) {
            for (int x = 0; x < grid_width; ++x) {
                const int location = level_offset + y * grid_width + x;
                const float score = output_value(location, 0);
                if (score < score_threshold) {
                    continue;
                }

                std::array<float, 4> distances{};
                for (int side = 0; side < 4; ++side) {
                    float maximum = -std::numeric_limits<float>::infinity();
                    for (int bin = 0; bin <= kRegMax; ++bin) {
                        maximum = std::max(maximum, output_value(location,
                            1 + side * (kRegMax + 1) + bin));
                    }
                    float denominator = 0.0F;
                    float numerator = 0.0F;
                    for (int bin = 0; bin <= kRegMax; ++bin) {
                        const float probability = std::exp(output_value(location,
                            1 + side * (kRegMax + 1) + bin) - maximum);
                        denominator += probability;
                        numerator += probability * static_cast<float>(bin);
                    }
                    distances[side] = numerator / denominator * static_cast<float>(stride);
                }

                const float center_x = (static_cast<float>(x) + 0.5F) * stride;
                const float center_y = (static_cast<float>(y) + 0.5F) * stride;
                const float left = std::clamp(center_x - distances[0], 0.0F, 320.0F);
                const float top = std::clamp(center_y - distances[1], 0.0F, 320.0F);
                const float right = std::clamp(center_x + distances[2], 0.0F, 320.0F);
                const float bottom = std::clamp(center_y + distances[3], 0.0F, 320.0F);
                candidates.push_back({
                    cv::Rect2f(left * original_size.width / kInputWidth,
                               top * original_size.height / kInputHeight,
                               (right - left) * original_size.width / kInputWidth,
                               (bottom - top) * original_size.height / kInputHeight),
                    score,
                });
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Detection& first, const Detection& second) {
        return first.score > second.score;
    });
    std::vector<Detection> detections;
    for (const auto& candidate : candidates) {
        if (detections.size() >= 100U) {
            break;
        }
        bool suppressed = false;
        for (const auto& kept : detections) {
            if (box_iou(candidate.box, kept.box) > kNmsThreshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            detections.push_back(candidate);
        }
    }
    return detections;
}

class OnnxDetector {
public:
    explicit OnnxDetector(const std::string& model_path)
        : environment_(ORT_LOGGING_LEVEL_WARNING, "nanodet"),
          session_options_(),
                    session_(environment_, model_path.c_str(), session_options_) {}

    std::vector<Detection> detect(const cv::Mat& frame, float threshold) {
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(kInputWidth, kInputHeight));
        std::vector<float> input_data(3U * kInputHeight * kInputWidth);
        for (int channel = 0; channel < 3; ++channel) {
            for (int y = 0; y < kInputHeight; ++y) {
                for (int x = 0; x < kInputWidth; ++x) {
                    input_data[static_cast<std::size_t>(channel * kInputHeight * kInputWidth
                        + y * kInputWidth + x)] = resized.at<cv::Vec3b>(y, x)[channel] / 255.0F;
                }
            }
        }

        const std::array<int64_t, 4> input_shape{1, 3, kInputHeight, kInputWidth};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(),
                                                              input_data.size(), input_shape.data(),
                                                              input_shape.size());
        const char* input_names[] = {"data"};
        const char* output_names[] = {"output"};
        auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                    output_names, 1);
        const float* output_data = outputs[0].GetTensorData<float>();
        const auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const std::size_t output_size = std::accumulate(output_shape.begin(), output_shape.end(),
                                                        std::size_t{1}, std::multiplies<>{});
        return decode_output(std::vector<float>(output_data, output_data + output_size),
                             frame.size(), threshold);
    }

private:
    Ort::Env environment_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
};

void draw_tracks(cv::Mat& frame, const std::vector<Track>& tracks) {
    for (const auto& track : tracks) {
        const auto& box = track.detection.box;
        const cv::Point top_left(static_cast<int>(box.x), static_cast<int>(box.y));
        const cv::Point bottom_right(static_cast<int>(box.x + box.width), static_cast<int>(box.y + box.height));
        cv::rectangle(frame, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "ID " + std::to_string(track.id),
                    cv::Point(top_left.x, std::max(0, top_left.y - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }
}

struct Arguments {
    std::string model = "nanodet_personOnly.onnx";
    std::string video = "video2.mp4";
    std::string output = "tracked_output.mp4";
    float threshold = 0.3F;
    bool camera = false;
    bool save = false;
    bool no_display = false;
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "  --model PATH       ONNX model (default: nanodet_personOnly.onnx)\n"
              << "  --video PATH       Input video (default: video2.mp4)\n"
              << "  --camera           Use webcam 0 instead of --video\n"
              << "  --threshold VALUE  Detection threshold (default: 0.3)\n"
              << "  --save PATH        Save annotated video to PATH\n"
              << "  --no-display       Disable the OpenCV preview window\n"
              << "  --help             Show this message\n";
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        auto value = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + option);
            }
            return argv[++index];
        };
        if (option == "--model") {
            arguments.model = value();
        } else if (option == "--video") {
            arguments.video = value();
        } else if (option == "--camera") {
            arguments.camera = true;
        } else if (option == "--threshold") {
            arguments.threshold = std::stof(value());
        } else if (option == "--save") {
            arguments.save = true;
            arguments.output = value();
        } else if (option == "--no-display") {
            arguments.no_display = true;
        } else if (option == "--help" || option == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown option: " + option);
        }
    }
    return arguments;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        OnnxDetector detector(arguments.model);

        cv::VideoCapture capture;
        if (arguments.camera) {
            capture.open(0);
        } else {
            capture.open(arguments.video);
        }
        if (!capture.isOpened()) {
            throw std::runtime_error("Could not open video source.");
        }

        cv::VideoWriter writer;
        MultiObjectTracker tracker(40.0F, 10, 1);
        cv::Mat frame;
        while (capture.read(frame)) {
            const auto start = cv::getTickCount();
            const auto detections = detector.detect(frame, arguments.threshold);
            const auto& tracks = tracker.update(detections);

            draw_tracks(frame, tracks);
            const double elapsed = (cv::getTickCount() - start) / cv::getTickFrequency();
            cv::putText(frame, "FPS: " + std::to_string(1.0 / std::max(elapsed, 1e-6)),
                        cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 0, 255), 2);

            if (arguments.save) {
                if (!writer.isOpened()) {
                    const double fps = capture.get(cv::CAP_PROP_FPS) > 0.0
                        ? capture.get(cv::CAP_PROP_FPS) : 30.0;
                    writer.open(arguments.output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                fps, frame.size());
                    if (!writer.isOpened()) {
                        throw std::runtime_error("Could not open output video.");
                    }
                }
                writer.write(frame);
            }

            if (!arguments.no_display) {
                cv::imshow("Object Tracking", frame);
                if ((cv::waitKey(1) & 0xFF) == 'q') {
                    break;
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}