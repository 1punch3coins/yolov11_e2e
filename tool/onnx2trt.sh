#!/bin/bash
echo "input image height: $1"
echo "input image width: $2"
echo "net input height: $3"
echo "net input width: $4"
echo "max batch size: 16"
echo "parse from $PWD/model/yolov11s_nx$1x$2x3($3x$4).onnx to $PWD/model/yolov11s_nx$1x$2x3($3x$4).trt"
echo "start to parse onnx model into trt model, it might consume minutes to finish; check log file at model/trt_build_f16.log"
/usr/src/tensorrt/bin/trtexec --onnx="./model/yolov11s_nx$1x$2x3($3x$4).onnx" --saveEngine="./model/yolov11s_nx$1x$2x3($3x$4).trt" --fp16 --plugins=build/libtrtrun_core.so --minShapes=input_img_uint8:1x$1x$2x3 --optShapes=input_img_uint8:4x$1x$2x3 --maxShapes=input_img_uint8:16x$1x$2x3 --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed > model/trt_build_f16.log 2>&1
echo "modle parse completed"