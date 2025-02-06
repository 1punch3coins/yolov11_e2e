#ifndef __TRT_RUNNER_V3_HPP__
#define __TRT_RUNNER_V3_HPP__

#include <vector>
#include <map>
#include <memory>

#include <NvInfer.h>
#include "cu_marcos.h"

class NetMeta {
public:
    enum class TensorType: int32_t {   // consistent with nvinfer1::DataType; use "enum class" to avoid naming conflict
        kTypeFloat32 = 0,
        kTypeFloat16 = 1,
        kTypeInt8    = 2,
        kTypeInt32   = 3,
        kTypeBool    = 4,
        kTypeUInt8   = 5,
        KTypeFloat8  = 6
    };
    static constexpr int32_t SizeTable[7] = {4, 2, 1, 4, 1, 1, 1};
    struct InputTensorMeta{
        std::string tensor_name;
        int32_t net_in_c;
        int32_t net_in_h;
        int32_t net_in_w;
        int32_t net_in_elems_num;
        TensorType tensor_type;
        bool in_nchw;
        InputTensorMeta(const std::string& tensor_name_, const TensorType& tensor_type_, const bool in_nchw_):
            tensor_name(std::move(tensor_name_)), tensor_type(tensor_type_), in_nchw(in_nchw_)
        {}
    };
    struct OutputTensorMeta{
        std::string tensor_name;
        int32_t net_out_c;
        int32_t net_out_l;
        int32_t net_out_h;
        int32_t net_out_w;
        int32_t net_out_elems_num;
        TensorType tensor_type;
        bool out_nlc;
        OutputTensorMeta(const std::string& tensor_name_, const TensorType& tensor_type_, const bool out_nlc_):
            tensor_name(std::move(tensor_name_)), tensor_type(tensor_type_), out_nlc(out_nlc_)
        {}
    };

public:
    int32_t batch_size;
    int32_t in_tensors_num;
    int32_t out_tensors_num;
    std::vector<InputTensorMeta> in_tensor_metas;
    std::vector<OutputTensorMeta> out_tensor_metas;
    std::map<std::string, int32_t> input_name2index;
    std::map<std::string, int32_t> output_name2index;

public:
    NetMeta(): 
        in_tensors_num(1),
        out_tensors_num(1)
    {}
    NetMeta(int32_t in_tensors_num_, int32_t out_tensors_num_):
        in_tensors_num(in_tensors_num_),
        out_tensors_num(out_tensors_num_)
    {}
    void AddInputTensorMeta (const std::string& input_name, const TensorType& tensor_type, const bool in_nchw) {
        in_tensor_metas.push_back(InputTensorMeta(input_name, tensor_type, in_nchw));
    }
    void AddOutputTensorMeta (const std::string& output_name, const TensorType& tensor_type, const bool out_nlc) {
        out_tensor_metas.push_back(OutputTensorMeta(output_name, tensor_type, out_nlc));
    }
};

class TrtRunnerV3 {
public:
    static TrtRunnerV3* Create();    // keyword static allowes it could be accessed via class name
    int32_t InitEngine(const std::string& model_pwd);
    int32_t InitMeta(NetMeta* t_info);
    int32_t InitBatchSize(const bool& is_dynamic, int32_t batch_size);
    int32_t InitMem();
    int32_t Initialize(const std::string& model_pwd, NetMeta* t_info, const int32_t& batch_size, const bool& is_dynamic);
    void Finalize();

public:
    void InferenceAsync(void* stream_ptr);
    void TransOutAsync(void* stream_ptr);
    void TransOutAsync(const std::string& tensor_name, void* stream_ptr);
    void SetCurrentBatchSize(int32_t const& batch_size);

public:
    void* GetDevInPtr(const std::string& tensor_name) const noexcept;
    const void* GetHostOutPtr(const std::string& tensor_name) const noexcept;
    const void* GetDevOutPtr(const std::string& tensor_name) const noexcept;
    int32_t GetInHeight(const std::string& tensor_name) const noexcept;
    int32_t GetInWidth(const std::string& tensor_name) const noexcept;
    int32_t GetOutLen(const std::string& tensor_name) const noexcept;
    int32_t GetOutHeight(const std::string& tensor_name) const noexcept;
    int32_t GetOutWidth(const std::string& tensor_name) const noexcept;
    int32_t GetOutChannelNum(const std::string& tensor_name) const noexcept;

private:
    int32_t dev_id_;
    std::string model_pwd_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

private:
    std::unique_ptr<NetMeta> net_meta_;

private:
    std::vector<void*> h_out_ptrs_;
    std::vector<void*> d_in_ptrs_;
    std::vector<void*> d_out_ptrs_;
    std::vector<void*> binding_ptrs_;
};

#endif