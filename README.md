# Multi-Object Tracking
* This repository is about implementation of `Kalman Filter` (an object tracking algorithm) with the light weight object detection model for the tracking of multiple objects.
* For this, i have trained the light weight one stage object detection [model](https://github.com/RangiLyu/nanodet) on coco2017 person only dataset.
* Here, i have done few changes in the original model as we can see in this directory.
* Locate this folder:
```bash
cd Object_Tracking
```
* Then, prepare the envorinment for CNN model, run following commands:
```bash
conda activate -n <env_name> python=3.8.10 -y
conda activate <env_name>
pip install -r requirements.txt
```

## Object Tracking Using python script
* To test the multi-object tracker using CNN object detection model and kalman filter algorithm, run following command:
```bash
python3 object_tracking.py --model <path to the .pth file> --vid_path <path to your video file>
```
* Result from python script: \
<video controls src="assets/tracked_output_python.mp4" title="python_result"></video>

## Object Tracking Using C++
* I have implemented python version of multi-object tracker in c++.
* So for this, first of all we will convert our pytorch model into `ONNX` format using following command:
```bash
python3 tools/export_onnx.py --model_path <path to the .pth file> --out_path model.onnx
```
* It will generate the `model.onnx` ONNX model.

### Now we need to make executable of the tracker:
* First we need onnxruntime library for linux, to get this:
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
tar -xzf onnxruntime-linux-x64-1.20.0.tgz
```
* Then, run following to make excutable file.
```bash
cmake -S . -B build -DONNXRUNTIME_ROOT=./onnxruntime-linux-x64-1.20.0
cmake --build build --parallel
```
* For the testing:
```
./build/object_tracking_cpp --model model.onnx --video video2.mp4
```

* Result from C++ script: \
<video controls src="assets/tracked_output_cpp.mp4" title="cpp_result"></video>

* Here,  we can see the difference in `fps` between the implementation of multi-object traker using python and using cpp after converting pytorch model into onnx format.

## References
* [Paper](https://arxiv.org/pdf/1710.04055).
* [Kalman Filter Expalanation](https://kalmanfilter.net/).

# Note
* I will update how to train the original light weight one stage detector [model](https://github.com/RangiLyu/nanodet) on single coco person only class and my own explanation about `Kalman Filter` in detail in this same repository later.