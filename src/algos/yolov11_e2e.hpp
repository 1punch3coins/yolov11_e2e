#ifndef YOLOV11_E2E_HPP_
#define YOLOV11_E2E_HPP_
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "trt_runner.hpp"
#include "img_precess_p.hpp"
#include "det_structs.h"

using PreParam = nvinfer1::plugin::PreParam;
using Crop = nvinfer1::plugin::Crop;
class Yolov11E2e {
public:
    struct Result {
        std::vector<std::vector<Bbox2D>> batched_bbox_vec;
        float process_time;
    };

public:
    Yolov11E2e(const std::string& labels_file="../config/label_coco_80.txt", const std::string& plugin_config="../config/plugin_config.yml"):
        labels_file_(labels_file), plugin_config_(plugin_config)
    {}

public:
    int32_t Initialize(const std::string& model);
    int32_t ReadClsNames(const std::string& filename);
    int32_t ReadCropConfigs(const std::string& filename);
    void Finalize(void);
    void Process(const cv::Mat& original_mat, Result& result);
    void Process(const std::vector<cv::Mat>& original_mat, Result& result);
    
private:
    std::unique_ptr<TrtRunner> trt_runner_;
    std::vector<std::string> cls_names_;
    cudaStream_t stream_;

private:
    Crop src_crop_;
    Crop dst_crop_;
    float scale_w_;
    float scale_h_;
    const std::string labels_file_;
    const std::string plugin_config_;
};

#endif