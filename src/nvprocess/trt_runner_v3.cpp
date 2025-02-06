#include <iostream>
#include <fstream>
#include <iterator>

#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include "trt_runner_v3.hpp"
#include "trt_logger.hpp"
#include "cu_marcos.h"

using Tt = NetMeta::TensorType;
static constexpr int32_t kLogInfoSize = 200;
static auto& logger = trtLogger::clogger.getTRTLogger();
TrtRunnerV3* TrtRunnerV3::Create() {
    TrtRunnerV3* p = new TrtRunnerV3();
    return p;
}

int32_t TrtRunnerV3::InitEngine(const std::string& model_pwd) {
    // 1. deseralize trt model and create corresponding engine and context
    model_pwd_ = model_pwd;
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    std::ifstream stream(model_pwd_, std::ios::binary);
    std::string buffer;
    if (stream) {
        stream >> std::noskipws;
        std::copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), back_inserter(buffer));   // copy file contents from ifstream to string using iterator
    }
    initLibNvInferPlugins(&logger, "");
    engine_.reset((runtime_->deserializeCudaEngine(buffer.data(), buffer.size())));
    context_.reset((engine_->createExecutionContext()));
    return 0;
}

int32_t TrtRunnerV3::InitMeta(NetMeta* p_meta) {
    // 2.1 readin model's input and output metadata
    net_meta_.reset(p_meta);
    char info[kLogInfoSize];
    int32_t input_meta_index = 0;
    int32_t output_meta_index = 0;
    
    if ((net_meta_->in_tensors_num + net_meta_->out_tensors_num) != engine_->getNbIOTensors()) {
        SET_ERROR_INFO(info, 1, "io tensors num does not match with model's");
        logger.log(trtLogger::Severity::kERROR, info);
        return 1;
    }
    for (auto& in_meta : net_meta_->in_tensor_metas) {
        const auto& tensor_name = in_meta.tensor_name.c_str();
        for (unsigned i = 0; i < engine_->getNbIOTensors(); i++) {
            if (strcmp(engine_->getIOTensorName(i),tensor_name) == 0) {break;}
            if (i == engine_->getNbIOTensors()-1) {
                SET_ERROR_INFO(info, 3, "could not find the input tensor", tensor_name, "in model");
                logger.log(trtLogger::Severity::kERROR, info);
                return 1;
            }
        }
        if (engine_->getTensorIOMode(tensor_name) != nvinfer1::TensorIOMode::kINPUT) {
            SET_ERROR_INFO(info, 2, tensor_name, "is not a input tensor");
            logger.log(trtLogger::Severity::kERROR, info);
            return 1;
        }
        if (int32_t(engine_->getTensorDataType(tensor_name)) != int32_t(in_meta.tensor_type)) {
            SET_ERROR_INFO(info, 3, "input tensor:", tensor_name, "'s datatype differs with model's setting");
            logger.log(trtLogger::Severity::kERROR, info);
            return 1;
        }
        nvinfer1::Dims input_shape = engine_->getTensorShape(tensor_name);
        if (in_meta.in_nchw) {
            in_meta.net_in_c = input_shape.d[1];
            in_meta.net_in_h = input_shape.d[2];
            in_meta.net_in_w = input_shape.d[3];
        } else {
            in_meta.net_in_h = input_shape.d[1];
            in_meta.net_in_w = input_shape.d[2];
            in_meta.net_in_c = input_shape.d[3];
        }
    }
    for (auto& out_meta : net_meta_->out_tensor_metas) {
        const auto& tensor_name = out_meta.tensor_name.c_str();
        for (unsigned i = 0; i < engine_->getNbIOTensors(); i++) {
            if (strcmp(engine_->getIOTensorName(i),tensor_name) == 0) {break;}
            if (i == engine_->getNbIOTensors()-1) {
                SET_ERROR_INFO(info, 3, "could not find the output tensor", tensor_name, "in model");
                logger.log(trtLogger::Severity::kERROR, info);
                return 1;
            }
        }
        if (engine_->getTensorIOMode(tensor_name) != nvinfer1::TensorIOMode::kOUTPUT) {
            SET_ERROR_INFO(info, 2, tensor_name, "is not a output tensor");
            logger.log(trtLogger::Severity::kERROR, info);
            return 1;
        }
        if (int32_t(engine_->getTensorDataType(tensor_name)) != int32_t(out_meta.tensor_type)) {
            SET_ERROR_INFO(info, 3, "output tensor:", tensor_name, "'s datatype differs with model's setting");
            logger.log(trtLogger::Severity::kERROR, info);
            return 1;
        }
        nvinfer1::Dims output_shape = engine_->getTensorShape(tensor_name);
        if (output_shape.nbDims == 2) {
            if (out_meta.out_nlc) {
                out_meta.net_out_l = output_shape.d[1];
                out_meta.net_out_c = 1;
            } else {
                out_meta.net_out_c = output_shape.d[1];
                out_meta.net_out_l = 1;
            }
        }
        if (output_shape.nbDims == 3) {
            if (out_meta.out_nlc) {
                out_meta.net_out_l = output_shape.d[1];
                out_meta.net_out_c = output_shape.d[2];
            } else {
                out_meta.net_out_c = output_shape.d[1];
                out_meta.net_out_l = output_shape.d[2];
            }
        }
        if (output_shape.nbDims == 4) {
            if (out_meta.out_nlc) {
                out_meta.net_out_h = output_shape.d[1];
                out_meta.net_out_w = output_shape.d[2];
                out_meta.net_out_c = output_shape.d[3];
                out_meta.net_out_l = output_shape.d[1] * output_shape.d[2];
            } else {
                out_meta.net_out_c = output_shape.d[1];
                out_meta.net_out_h = output_shape.d[2];
                out_meta.net_out_w = output_shape.d[3];
                out_meta.net_out_l = output_shape.d[2] * output_shape.d[3];
            }
        }
    }

    for (int i = 0; i < net_meta_->in_tensor_metas.size(); i++) {
        net_meta_->input_name2index.insert({net_meta_->in_tensor_metas[i].tensor_name, i});
    }
    for (int i = 0; i < net_meta_->out_tensor_metas.size(); i++) {
        net_meta_->output_name2index.insert({net_meta_->out_tensor_metas[i].tensor_name, i});
    }
    return 0;
}

int32_t TrtRunnerV3::InitBatchSize(const bool& is_dynamic, int32_t batch_size) {
    // 3. check model's batch size and calc tensors sizes
    char info[kLogInfoSize];
    if (is_dynamic) {
        for (int32_t i = 0; i < net_meta_->in_tensors_num; i++) {
            const auto& tensor_name = net_meta_->in_tensor_metas[i].tensor_name.c_str();
            const nvinfer1::Dims& input_shape = engine_->getTensorShape(tensor_name);
            if (input_shape.d[0] != -1) {
                SET_ERROR_INFO(info, 1, "model is not of dyanmic batch size");
                logger.log(trtLogger::Severity::kERROR, info);
                return 1;
            }
            nvinfer1::Dims max_shape = engine_->getProfileShape(tensor_name, 0, nvinfer1::OptProfileSelector::kMAX);
            if (max_shape.d[0] < batch_size) {
                SET_ERROR_INFO(info, 1, "model's batch size is out of allowed range");
                logger.log(trtLogger::Severity::kERROR, info);
                return 1;
            }
        }
    } else {
        for (auto& in_meta : net_meta_->in_tensor_metas) {
            const auto& tensor_name = in_meta.tensor_name.c_str();
            const nvinfer1::Dims& input_shape = engine_->getTensorShape(tensor_name);
            if (input_shape.d[0] != batch_size) {
                SET_ERROR_INFO(info, 4, "model's tensor", in_meta.tensor_name.c_str(), "batch size is", std::to_string(input_shape.d[0]).c_str());
                logger.log(trtLogger::Severity::kERROR, info);
                return 1;
            }
        }
    }
    net_meta_->batch_size = batch_size;
    for (auto& in_meta : net_meta_->in_tensor_metas) {
        in_meta.net_in_elems_num = net_meta_->batch_size * in_meta.net_in_c * in_meta.net_in_h * in_meta.net_in_w;
    }
    for (auto& out_meta : net_meta_->out_tensor_metas) {
        out_meta.net_out_elems_num = net_meta_->batch_size * out_meta.net_out_c * out_meta.net_out_l;
    }
    return 0;    
}

int32_t TrtRunnerV3::InitMem() {
    // 4. construct network's input and output mem space on host and device
    h_out_ptrs_.resize(net_meta_->out_tensors_num);
    d_in_ptrs_.resize(net_meta_->in_tensors_num);
    d_out_ptrs_.resize(net_meta_->out_tensors_num);
    for (int32_t i = 0; i < net_meta_->in_tensors_num; i++) {
        const auto& in_meta = net_meta_->in_tensor_metas[i];
        CHECK_CUDA_FUNC(cudaMalloc((void**)&d_in_ptrs_[i], in_meta.net_in_elems_num*NetMeta::SizeTable[int32_t(in_meta.tensor_type)]));
        binding_ptrs_.push_back(d_in_ptrs_[i]);
        bool status = context_->setInputTensorAddress(in_meta.tensor_name.c_str(), d_in_ptrs_[i]);
        if (!status) {return 1;}
    }
    for (int32_t i = 0; i < net_meta_->out_tensors_num; i++) {
        const auto& out_meta = net_meta_->out_tensor_metas[i];
        CHECK_CUDA_FUNC(cudaMallocHost((void**)&h_out_ptrs_[i], out_meta.net_out_elems_num*NetMeta::SizeTable[int32_t(out_meta.tensor_type)], cudaHostAllocDefault));
        CHECK_CUDA_FUNC(cudaMalloc((void**)&d_out_ptrs_[i], out_meta.net_out_elems_num*NetMeta::SizeTable[int32_t(out_meta.tensor_type)]));
        bool status = context_->setOutputTensorAddress(out_meta.tensor_name.c_str(), d_out_ptrs_[i]);
        if (!status) {return 1;}
    }
    return 0;
}

int32_t TrtRunnerV3::Initialize(const std::string& model_pwd, NetMeta* t_info, const int32_t& batch_size, const bool& is_dynamic) {
    if (!InitEngine(model_pwd)) {return 1;}
    if (!InitMeta(t_info)) {return 1;}
    if (!InitBatchSize(is_dynamic, batch_size)) {return 1;}
    if (!InitMem()) {return 1;}
    return 0;
}

void TrtRunnerV3::Finalize() {
    for (auto& pinned_ptr : h_out_ptrs_) {
        CHECK_CUDA_FUNC(cudaFreeHost(pinned_ptr));
    }
    for (auto& d_ptr : d_in_ptrs_) {
        CHECK_CUDA_FUNC(cudaFree(d_ptr));
    }
    for (auto& d_ptr : d_out_ptrs_) {
        CHECK_CUDA_FUNC(cudaFree(d_ptr));
    }
}

void TrtRunnerV3::InferenceAsync(void* stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    context_->enqueueV3(stream);
}

void TrtRunnerV3::TransOutAsync(void* stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    for (int32_t i = 0; i < net_meta_->out_tensors_num; i++) {
        const auto& out_meta = net_meta_->out_tensor_metas[i];
        CHECK_CUDA_FUNC(cudaMemcpyAsync(h_out_ptrs_[i], d_out_ptrs_[i], out_meta.net_out_elems_num * NetMeta::SizeTable[int32_t(out_meta.tensor_type)], cudaMemcpyDeviceToHost, stream));
    }
}

void TrtRunnerV3::TransOutAsync(const std::string& tensor_name, void* stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    int32_t index = net_meta_->output_name2index.find(tensor_name)->second;
    const auto& out_meta = net_meta_->out_tensor_metas[index];
    CHECK_CUDA_FUNC(cudaMemcpyAsync(h_out_ptrs_[index], d_out_ptrs_[index], out_meta.net_out_elems_num * NetMeta::SizeTable[int32_t(out_meta.tensor_type)], cudaMemcpyDeviceToHost, stream));
}

void TrtRunnerV3::SetCurrentBatchSize(int32_t const& batch_size) {
    if (batch_size > net_meta_->batch_size) {
        char info[kLogInfoSize];
        SET_ERROR_INFO(info, 4, "current batch size", std::to_string(batch_size).c_str(), "exceeds model's allowed batch size", net_meta_->batch_size);
        logger.log(trtLogger::Severity::kERROR, info);
    }
    for (int32_t i = 0; i < net_meta_->in_tensors_num; i++) {
        const auto& in_meta = net_meta_->in_tensor_metas[i];
        if (in_meta.in_nchw) {
            context_->setInputShape(in_meta.tensor_name.c_str(), nvinfer1::Dims4(batch_size, in_meta.net_in_c, in_meta.net_in_h, in_meta.net_in_w));
        } else {
            context_->setInputShape(in_meta.tensor_name.c_str(), nvinfer1::Dims4(batch_size, in_meta.net_in_h, in_meta.net_in_w, in_meta.net_in_c));
        }
    }
    if (!context_->allInputDimensionsSpecified()) {
        char info[kLogInfoSize];
        SET_ERROR_INFO(info, 1, "not all requried diemensions set");
        logger.log(trtLogger::Severity::kERROR, info);
    }
    for (auto& out_meta : net_meta_->out_tensor_metas) {
        out_meta.net_out_elems_num = batch_size * out_meta.net_out_c * out_meta.net_out_l;
    }
}

inline static void LogNotFoundErr(const std::string& tensor_name) {
    char info[kLogInfoSize];
    SET_ERROR_INFO(info, 2, "could not find the tensor", tensor_name.c_str());
    logger.log(trtLogger::Severity::kERROR, info);
}

void* TrtRunnerV3::GetDevInPtr(const std::string& tensor_name) const noexcept {
    auto iter = net_meta_->input_name2index.find(tensor_name);
    if (iter != net_meta_->input_name2index.end()) {
        return d_in_ptrs_[iter->second];
    } else {
        LogNotFoundErr(tensor_name);
        exit(1);
    }
}
const void* TrtRunnerV3::GetHostOutPtr(const std::string& tensor_name) const noexcept {
    auto iter = net_meta_->output_name2index.find(tensor_name);
    if (iter != net_meta_->output_name2index.end()) {
        return h_out_ptrs_[iter->second];
    } else {
        LogNotFoundErr(tensor_name);
        exit(1);
    }
}
const void* TrtRunnerV3::GetDevOutPtr(const std::string& tensor_name) const noexcept {
    auto iter = net_meta_->output_name2index.find(tensor_name);
    if (iter != net_meta_->output_name2index.end()) {
        return d_out_ptrs_[iter->second];
    } else {
        LogNotFoundErr(tensor_name);
        exit(1);
    }
}
int32_t TrtRunnerV3::GetInHeight(const std::string& tensor_name) const noexcept {
    auto iter = net_meta_->input_name2index.find(tensor_name);
    if (iter != net_meta_->input_name2index.end()) {
        return net_meta_->in_tensor_metas[iter->second].net_in_h;
    } else {
        LogNotFoundErr(tensor_name);
        exit(1);
    }
}
int32_t TrtRunnerV3::GetInWidth(const std::string& tensor_name) const noexcept {
    auto iter = net_meta_->input_name2index.find(tensor_name);
    if (iter != net_meta_->input_name2index.end()) {
        return net_meta_->in_tensor_metas[iter->second].net_in_w;
    } else {
        LogNotFoundErr(tensor_name);
        exit(1);
    }
}
int32_t TrtRunnerV3::GetOutLen(const std::string& tensor_name) const noexcept {
    auto iter = net_meta_->output_name2index.find(tensor_name);
    if (iter != net_meta_->output_name2index.end()) {
        return net_meta_->out_tensor_metas[iter->second].net_out_l;
    } else {
        LogNotFoundErr(tensor_name);
        exit(1);
    }
}
int32_t TrtRunnerV3::GetOutHeight(const std::string& tensor_name) const noexcept {
    auto iter = net_meta_->output_name2index.find(tensor_name);
    if (iter != net_meta_->output_name2index.end()) {
        return net_meta_->out_tensor_metas[iter->second].net_out_h;
    } else {
        LogNotFoundErr(tensor_name);
        exit(1);
    }
}
int32_t TrtRunnerV3::GetOutWidth(const std::string& tensor_name) const noexcept {
    auto iter = net_meta_->output_name2index.find(tensor_name);
    if (iter != net_meta_->output_name2index.end()) {
        return net_meta_->out_tensor_metas[iter->second].net_out_w;
    } else {
        LogNotFoundErr(tensor_name);
        exit(1);
    }
}
int32_t TrtRunnerV3::GetOutChannelNum(const std::string& tensor_name) const noexcept {
    auto iter = net_meta_->output_name2index.find(tensor_name);
    if (iter != net_meta_->output_name2index.end()) {
        return net_meta_->out_tensor_metas[iter->second].net_out_c;
    } else {
        LogNotFoundErr(tensor_name);
        exit(1);
    }
}