#include <iostream>
#include <fstream>
#include <chrono>
#include <array>
#include <algorithm>
#include "yaml-cpp/yaml.h"
#include "algo_logger.hpp"
#include "yolov11_e2e.hpp"

#if DYNAMIC_BATCH_SIZE
static constexpr int32_t kBatchSize = 16;
#else
static constexpr int32_t kBatchSize = 1;
#endif

// model's input's and output's meta
using Tt = NetMeta::TensorType;
static constexpr int32_t kInputTensorNum = 1;
static constexpr int32_t kOutputTensorNum = 4;
static constexpr std::array<const char*, kInputTensorNum> sInputNameList = {"input_img_uint8"};
static constexpr std::array<const Tt, kInputTensorNum> kInputTypeList = {Tt::kTypeUInt8};
static constexpr std::array<const bool, kOutputTensorNum> iInputNchwList = {false};
static constexpr std::array<const char*, kOutputTensorNum> sOutputNameList = {"num", "boxes", "classes", "scores"};
static constexpr std::array<const bool, kOutputTensorNum> iOutputNlcList = {true, true, true, true};
static constexpr std::array<const Tt, kOutputTensorNum> kOutputTypeList = {Tt::kTypeInt32, Tt::kTypeFloat32, Tt::kTypeInt32, Tt::kTypeFloat32};
// used for post_process and net output meta check
static constexpr std::array<int32_t, kOutputTensorNum> kOutputChannelList = {1, 4, 1, 1};
static constexpr int32_t kOutputLen = 100;
static constexpr int32_t kClassNum = 80;
static constexpr auto& kBoxNumChannelNum = kOutputChannelList[0];
static constexpr auto& kOutputBoxChannelNum = kOutputChannelList[1];
static constexpr auto& kOutputConfChannelNum = kOutputChannelList[2];
static constexpr auto& kOutputIdChannelNum = kOutputChannelList[3];
// used for logger
static constexpr const char* sIdentifier = {"YOLOV11"};
static constexpr int32_t kLogInfoSize = 100;
static auto logger = algoLogger::logger;

int32_t Yolov11E2e::Initialize(const std::string& model) {
    // set net meta from pre_config
    LOG_INFO(logger, sIdentifier, "initializing...");
    NetMeta* p_meta = new NetMeta(kInputTensorNum, kOutputTensorNum);
    for (int32_t i = 0; i < kInputTensorNum; i++) {
        p_meta->AddInputTensorMeta(sInputNameList[i], kInputTypeList[i], iInputNchwList[i]);
    }
    for (int32_t i = 0; i < kOutputTensorNum; i++) {
        p_meta->AddOutputTensorMeta(sOutputNameList[i], kOutputTypeList[i], iOutputNlcList[i]);
    }
#if USE_ENQUEUEV3
    trt_runner_.reset(TrtRunnerV3::Create());
#else
    trt_runner_.reset(TrtRunner::Create());
#endif

    // create model and set net meta values from engine
    if (trt_runner_->InitEngine(model)) {
        trt_runner_.reset();
        LOG_ERROR_RETURN(logger, sIdentifier, "engine creation failed");
    }
    LOG_INFO(logger, sIdentifier, "model's engine creation completed");
    if (trt_runner_->InitMeta(p_meta)) {
        trt_runner_.reset();
        LOG_ERROR_RETURN(logger, sIdentifier, "net meta initialization failed");
    }
    LOG_INFO(logger, sIdentifier, "net meta initialization completed");
    if (trt_runner_->InitBatchSize(DYNAMIC_BATCH_SIZE, kBatchSize)) {
        trt_runner_.reset();
        LOG_ERROR_RETURN(logger, sIdentifier, "batch size initialization failed");
        
    }
    LOG_INFO(logger, sIdentifier, "batch size initialization completed");
    if (trt_runner_->InitMem()) {
        trt_runner_.reset();
        LOG_ERROR_RETURN(logger, sIdentifier, "model's memory initialization failed");
    }
    LOG_INFO(logger, sIdentifier, "model's memory initialization completed");

    // check output tensor meta
    for (unsigned i = 0; i < sOutputNameList.size(); i++) {
        if (trt_runner_->GetOutChannelNum(sOutputNameList[i]) != kOutputChannelList[i]) {
            trt_runner_.reset();
            LOG_ERROR_RETURN(logger, sIdentifier, "output channel size mismatched");
        }
    }
    for (unsigned i = 1; i < sOutputNameList.size(); i++) {
        if (trt_runner_->GetOutLen(sOutputNameList[i]) != kOutputLen) {
            trt_runner_.reset();
            LOG_ERROR_RETURN(logger, sIdentifier, "output len mismatched");
        }
    }
    LOG_INFO(logger, sIdentifier, "output meta check completed");

    // read label and plugin config files
    if(ReadClsNames(labels_file_))  {
        trt_runner_.reset();
        LOG_ERROR_RETURN(logger, sIdentifier, "failed to read labels file");
    }
    if(cls_names_.size() != kClassNum) {
        trt_runner_.reset();
        LOG_ERROR_RETURN(logger, sIdentifier, "lable size does not match net's output class num");
    }
    if (ReadCropConfigs(plugin_config_)) {
        trt_runner_.reset();
        LOG_ERROR_RETURN(logger, sIdentifier, "failed to read crop config file");
    }

    // create cuda stream
    CHECK_CUDA_FUNC(cudaStreamCreate(&stream_));
    LOG_INFO(logger, sIdentifier, "initialization completed");
    return 0;
}

int32_t Yolov11E2e::ReadClsNames(const std::string& filename) {
    std::ifstream ifs(filename);
    if (ifs.fail()) {
        return 1;
    }
    cls_names_.clear();
    std::string str;
    while (getline(ifs, str)) {
        cls_names_.push_back(str);
    }
    return 0;
}

int32_t Yolov11E2e::ReadCropConfigs(const std::string& filename) {
    //auto logger = algologger::logger;
    YAML::Node config;
    try {
        config = YAML::LoadFile(filename);
    } catch (YAML::Exception& e) {
        return 1;
    }
    if (config["precess_plugin"]) {
        auto precess_config = config["precess_plugin"];
        if (precess_config["src_crop"]) {
            auto src_crop_config = precess_config["src_crop"];
            if (src_crop_config.size() != 4) {
                LOG_ERROR_RETURN(logger, sIdentifier, "item \"src_crop\"'s length is not 4");
            }
            src_crop_.l = src_crop_config[0].as<unsigned>();
            src_crop_.t = src_crop_config[1].as<unsigned>();
            src_crop_.w = src_crop_config[2].as<unsigned>();
            src_crop_.h = src_crop_config[3].as<unsigned>();
        } else {
            LOG_ERROR_RETURN(logger, sIdentifier, "coud not find item of \"src_crop\"");
        }
        if (precess_config["dst_crop"]) {
            auto dst_crop_config = precess_config["dst_crop"];
            if (dst_crop_config.size() != 4) {
                LOG_ERROR_RETURN(logger, sIdentifier, "item \"dst_crop\"'s length is not 4");
            }
            dst_crop_.l = dst_crop_config[0].as<unsigned>();
            dst_crop_.t = dst_crop_config[1].as<unsigned>();
            dst_crop_.w = dst_crop_config[2].as<unsigned>();
            dst_crop_.h = dst_crop_config[3].as<unsigned>();
        } else {
            LOG_ERROR_RETURN(logger, sIdentifier, "coud not find item of \"dst_crop\"");
        }
        if (precess_config["scale_inv"]) {
            auto scale_config = precess_config["scale_inv"];
            assert(scale_config.size() == 2);
            scale_w_ = scale_config[0].as<float>();
            scale_h_ = scale_config[1].as<float>();
        } else {
            LOG_ERROR_RETURN(logger, sIdentifier, "coud not find item of \"scale_inv\"");
        }
    } else {
        LOG_ERROR_RETURN(logger, sIdentifier, "coud not find item of \"precess_plugin\"");
    }
    return 0;
}

void Yolov11E2e::Process(const cv::Mat& cv_mat, Result& result) {
    // 1. prep-rocess
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    const std::string& input_name = sInputNameList[0];
    cudaMemcpyAsync(trt_runner_->GetDevInPtr(input_name), cv_mat.data, cv_mat.cols*cv_mat.rows*3*sizeof(uint8_t), cudaMemcpyHostToDevice, stream_);
    CHECK_CUDA_FUNC(cudaStreamSynchronize(stream_));
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    // 2. inference
    trt_runner_->SetCurrentBatchSize(1);
    trt_runner_->InferenceAsync(stream_);
    CHECK_CUDA_FUNC(cudaStreamSynchronize(stream_));
    const auto& t_infer = std::chrono::steady_clock::now();
    
    trt_runner_->TransOutAsync(stream_);
    CHECK_CUDA_FUNC(cudaStreamSynchronize(stream_));
    const auto& t_post_process0 = std::chrono::steady_clock::now();

    // 3. post-process, retrive output; scale and offset bboxes to input img
    const unsigned dets_num = *(int*)trt_runner_->GetHostOutPtr(sOutputNameList[0]);
    const float* boxes      = (float*)trt_runner_->GetHostOutPtr(sOutputNameList[1]);
    const int* cls_ids      = (int*)trt_runner_->GetHostOutPtr(sOutputNameList[2]);
    const float* cls_scores = (float*)trt_runner_->GetHostOutPtr(sOutputNameList[3]);
    result.batched_bbox_vec.resize(1);
    std::vector<Bbox2D>& bbox_vec = result.batched_bbox_vec[0];
    bbox_vec.reserve(dets_num);
    for (unsigned i = 0; i < dets_num; i++) {
        int cls_id = cls_ids[i];
        float cls_conf = cls_scores[i];
        unsigned index = i*kOutputBoxChannelNum;
        int x0 = static_cast<int>((boxes[index + 0] - dst_crop_.l) * scale_w_) + src_crop_.l;
        int y0 = static_cast<int>((boxes[index + 1] - dst_crop_.t) * scale_h_) + src_crop_.t;
        int x1 = static_cast<int>((boxes[index + 2] - dst_crop_.l) * scale_w_) + src_crop_.l;
        int y1 = static_cast<int>((boxes[index + 3] - dst_crop_.t) * scale_h_) + src_crop_.t;
        int w = static_cast<int>(x1 - x0);
        int h = static_cast<int>(y1 - y0);
        std::string cls_name = cls_names_[cls_id];
        bbox_vec.push_back(Bbox2D(cls_id, cls_name, cls_conf, x0, y0, w, h));
    }
    const auto& t_post_process1 = std::chrono::steady_clock::now();
    result.process_time = (t_post_process1 - t_pre_process0).count() * 1e-6f;
#ifdef PRINT_TIMING
    char infos[5][kLogInfoSize];
    //auto logger = algologger::logger;
    std::cout << "---------------------\n";
    SET_TIMING_INFO(infos[0], "host_to_device", (t_pre_process1-t_pre_process0).count()*1e-6f, "ms");
    SET_TIMING_INFO(infos[1], "inference     ", (t_infer-t_pre_process1).count()*1e-6f, "ms");
    SET_TIMING_INFO(infos[2], "device_to_host", (t_post_process0-t_infer).count()*1e-6f, "ms");
    SET_TIMING_INFO(infos[3], "post-process  ", (t_post_process1-t_post_process0).count()*1e-6f, "ms");
    SET_TIMING_INFO(infos[4], "total         ", result.process_time, "ms");
    LOG_INFO(logger, sIdentifier, infos[0]);
    LOG_INFO(logger, sIdentifier, infos[1]);
    LOG_INFO(logger, sIdentifier, infos[2]);
    LOG_INFO(logger, sIdentifier, infos[3]);
    LOG_INFO(logger, sIdentifier, infos[4]);
#endif
}

void Yolov11E2e::Process(const std::vector<cv::Mat>& cv_mat_vec, Result& result) {
    // 1. prep-rocess
    //auto logger = algologger::logger;
    const int batch_size = cv_mat_vec.size();
#if DYNAMIC_BATCH_SIZE
    if (batch_size > kBatchSize) {
        LOG_ERROR(logger, sIdentifier, "input's batch size exceeds model's capacity");
        exit(1);
    }
#else
    if (batch_size != kBatchSize) {
        LOG_ERROR(logger, sIdentifier, "input's batch size does not match model's setting");
        exit(1);
    }
#endif
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    const std::string& input_name = sInputNameList[0];
    auto devin_ptr = (uint8_t*)trt_runner_->GetDevInPtr(input_name);
    for (const auto& cv_mat : cv_mat_vec) {
        cudaMemcpyAsync(devin_ptr, cv_mat.data, cv_mat.cols*cv_mat.rows*3*sizeof(uint8_t), cudaMemcpyHostToDevice, stream_);
        devin_ptr += cv_mat.cols*cv_mat.rows*3;
    }
    CHECK_CUDA_FUNC(cudaStreamSynchronize(stream_));
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    // 2. inference
    trt_runner_->SetCurrentBatchSize(batch_size);
    trt_runner_->InferenceAsync(stream_);
    CHECK_CUDA_FUNC(cudaStreamSynchronize(stream_));
    const auto& t_infer = std::chrono::steady_clock::now();
    
    trt_runner_->TransOutAsync(stream_);
    CHECK_CUDA_FUNC(cudaStreamSynchronize(stream_));
    const auto& t_post_process0 = std::chrono::steady_clock::now();

    // 3. post-process, retrive output; scale and offset bboxes to input img
    const unsigned* dets_num_ptr   = (unsigned*)trt_runner_->GetHostOutPtr(sOutputNameList[0]);
    const float*    boxes_ptr      = (float*)trt_runner_->GetHostOutPtr(sOutputNameList[1]);
    const int*      cls_ids_ptr    = (int*)trt_runner_->GetHostOutPtr(sOutputNameList[2]);
    const float*    cls_scores_ptr = (float*)trt_runner_->GetHostOutPtr(sOutputNameList[3]);
    result.batched_bbox_vec.resize(batch_size);
    for (unsigned i = 0; i < batch_size; i++) {
        unsigned dets_num = *dets_num_ptr;
        std::vector<Bbox2D>& bbox_vec = result.batched_bbox_vec[i];
        bbox_vec.reserve(dets_num);
        for (unsigned i = 0; i < dets_num; i++) {
            int cls_id = cls_ids_ptr[i];
            float cls_conf = cls_scores_ptr[i];
            unsigned box_index = i*kOutputBoxChannelNum;
            int x0 = static_cast<int>((boxes_ptr[box_index + 0] - dst_crop_.l) * scale_w_) + src_crop_.l;
            int y0 = static_cast<int>((boxes_ptr[box_index + 1] - dst_crop_.t) * scale_h_) + src_crop_.t;
            int x1 = static_cast<int>((boxes_ptr[box_index + 2] - dst_crop_.l) * scale_w_) + src_crop_.l;
            int y1 = static_cast<int>((boxes_ptr[box_index + 3] - dst_crop_.t) * scale_h_) + src_crop_.t;
            int w = static_cast<int>(x1 - x0);
            int h = static_cast<int>(y1 - y0);
            std::string cls_name = cls_names_[cls_id];
            bbox_vec.push_back(Bbox2D(cls_id, cls_name, cls_conf, x0, y0, w, h));
        }
        dets_num_ptr++;
        cls_ids_ptr    += kOutputLen*kOutputIdChannelNum;
        cls_scores_ptr += kOutputLen*kOutputConfChannelNum;
        boxes_ptr      += kOutputLen*kOutputBoxChannelNum;
    }
    const auto& t_post_process1 = std::chrono::steady_clock::now();
    result.process_time = 1.0 * (t_post_process1 - t_pre_process0).count() * 1e-6;
#ifdef PRINT_TIMING
    std::cout << "---------------------\n";
    char infos[5][kLogInfoSize];
    SET_TIMING_INFO(infos[0], "host_to_device", (t_pre_process1-t_pre_process0).count()*1e-6, "ms");
    SET_TIMING_INFO(infos[1], "inference     ", (t_infer-t_pre_process1).count()*1e-6, "ms");
    SET_TIMING_INFO(infos[2], "device_to_host", (t_post_process0-t_infer).count()*1e-6, "ms");
    SET_TIMING_INFO(infos[3], "post-process  ", (t_post_process1-t_post_process0).count()*1e-6, "ms");
    SET_TIMING_INFO(infos[4], "total         ", result.process_time, "ms");
    LOG_INFO(logger, sIdentifier, infos[0]);
    LOG_INFO(logger, sIdentifier, infos[1]);
    LOG_INFO(logger, sIdentifier, infos[2]);
    LOG_INFO(logger, sIdentifier, infos[3]);
    LOG_INFO(logger, sIdentifier, infos[4]);
#endif
}

void Yolov11E2e::Finalize() {
    trt_runner_->Finalize();
    trt_runner_.reset();
    CHECK_CUDA_FUNC(cudaStreamDestroy(stream_));
}