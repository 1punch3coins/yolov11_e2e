# End-to-end yolo11
The project is a tensorrt-based, end-to-end implemntation of yolo11 in c++ environment.
## ⚡ Features
- ✅ End-to-end model implemntation with preprocess and postprocess integrated.
- ✅ Both tensorrt's enqueueV2 and enqueueV3 apis are supported.
- ✅ Dynamic batch size.
- ✅ Fine-granularity logs to test performance.
## 📌 Table of Contents
- [Environment](#-environment)
- [Usage](#-usage)
- [Build-your-own-model](#-build-your-own-model)
- [License](#-license)
## 📝 Environment
The project has been tested under following enviroments.
- ✅ Ubuntu18.04, gcc7.5, cuda11.4, tensorrt8.6.1
- ✅ Ubuntu22.05, gcc11.4, cuda12.2, tensorrt10.4

Currently Windows is not supported. Other configurations should also work under Ubuntu.
## 📖 Usage
### 0. Envrioment check.
Check the above envrioment requriments and make sure opencv >= 4.0 is installed.
### 1. Clone this repo.
```bash
git clone https://github.com/1punch3coins/yolov11_e2e.git
```
### 2. Build the files.
```bash
cd path/to/repo
mkdir build && cd build
cmake ../ && make -j
```
### 3. Parse onnx model into tensorrt. After minutes of parsing, you should see a built .trt file and building log under model folder.
```bash
cd ..
bash tool/onnx2trt.sh
```
### 4. Launch the demo using "./yolo11 path/to/model  folder/of/inputs  folder/of/outputs".
```bash
cd build
./yolo11 "../model/yolov11s_nx720x1280x3(384x640).trt" "../assets/input" "../assets/output"
```
### 5. Check results and logs.
You should see some logs on the terminal like followings. And you might find the inference time is high as device needs a warm-up to fully utilize its resource. Also note that there is still a post-process procedure to rescale boxes into its input image sizes.
```
[12/22/2024-19:03:48] [YOLOV11] initializing...
[12/22/2024-19:03:48] [I] [TRT] Loaded engine size: 22 MiB
[12/22/2024-19:03:48] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +231, now: CPU 0, GPU 249 (MiB)
[12/22/2024-19:03:48] [YOLOV11] model's engine creation completed
[12/22/2024-19:03:48] [YOLOV11] net meta initialization completed
[12/22/2024-19:03:48] [YOLOV11] batch size initialization completed
[12/22/2024-19:03:48] [YOLOV11] model's memory initialization completed
[12/22/2024-19:03:48] [YOLOV11] output meta check completed
[12/22/2024-19:03:48] [YOLOV11] initialization completed
---------------------
[12/22/2024-19:03:49] [YOLOV11] host_to_device: 0.657921 ms
[12/22/2024-19:03:49] [YOLOV11] inference     : 83.270593 ms
[12/22/2024-19:03:49] [YOLOV11] device_to_host: 0.018905 ms
[12/22/2024-19:03:49] [YOLOV11] post-process  : 0.010100 ms
[12/22/2024-19:03:49] [YOLOV11] total         : 83.957520 ms
```
## 🛠️ Build-Your-Own-Model
### 0. Motivation:
There are two plugins attached to the nn part of yolo11 model, preprocessing input image and postprocessing nn model's output dens boxes respectively. The preprocess plugin is implmented with cuda kernels in src/nvprocess/img_precess_k.cu, while nms plugin is from tensorrt's official plugin libs.

According to [here](https://docs.nvidia.com/deeplearning/tensorrt/10.8.0/architecture/capabilities.html#), tensorrt doesn't support uint8 data format to do internal computation, so the float/half conversion is explicitly done in a precedent cast node, instead in the preprocess plugin node.
| ImgPrecessPlugin | EfficientNMS_TRT |
|---------|---------|
| ![Alt1](./assets/docs/Screenshot%20from%202025-02-06%2022-20-27.png) | ![Alt2](./assets/docs/Screenshot%20from%202025-02-06%2022-20-37.png) |

You can find the preprocess plugin does not support dynamic input image size, the repo only supports input image size of 720x1280. If you want the model to deal with another certain resolution, you should build your own model as the following.
### 1. Set image preprocess plugin's properties
```bash
ih=your_image_height    # your image input resolution
iw=your_image_width
nih=nn_input_height     # your nn part's input size, it effects inference result's accuracy and inference time
niw=nn_input_width      # if you don't know which size is suitable, just set as 640x640
python tool/gen_precess_config.py --input_h $ih --input_w $iw --net_input_h $nih --net_input_w $niw
```
Note "ih" and "iw" is your input image' resolution, while "nih" and "niw" is the size of nn's input. You should set these variables yourself accroding to your needs. This script would output config options into config/plugin_config.yml.
### 2. Clone yolo11 from official repo and prepare python environment.
```bash
git clone https://github.com/ultralytics/ultralytics.git
yolo11_path=$(pwd)/ultralytics  # record yolo11 repo's path
pip3 install ultralytics thop onnx onnxsim onnx_graphsurgeon
```
### 3. Export onnx model from pytorch's .pt file.
```bash
cp tool/export_e2e_onnx.py $yolo11_path/
base_path=$(pwd)
cd $yolo11_path
python export_e2e_onnx.py --weights $base_path/model/yolo11s.pt --output $base_path/model --config $base_path/config/plugin_config.yml  --input_h $ih --input_w $iw --net_input_h $nih --net_input_w $niw --end2end --dynamic_batch_size
cd $base_path
```
### 4. Export trt model from onnx model
```bash
bash tool/onnx2trt.sh $ih $iw $nih $niw
```
## 📜 License
The repo is under MIT license.
## 🔍 References
🔗 Yolo11: https://github.com/ultralytics/ultralytics.git  
🔗 Tensorrt'nms plugin: https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin/README.md