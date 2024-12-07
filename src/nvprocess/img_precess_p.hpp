#pragma once
#include <vector>
#include <string>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <NvInferPlugin.h>

namespace nvinfer1 {

namespace plugin {

struct half3 {
    half x;
    half y;
    half z;
};

struct Crop {
    unsigned l;
    unsigned t;
    unsigned w;
    unsigned h;
};

struct PreParam {
    unsigned batch_size;
    Crop src_crop;
    Crop dst_crop;
    uint2 src_size;
    uint2 dst_size;
    unsigned src_step;
    unsigned dst_step;
    float2 scale_inv;
    float3 mean;
    float3 norm_inv;
    half3 mean16;
    half3 norm_inv16;
};

class ImgPrecessPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    ImgPrecessPlugin() = delete; // disable the default constructor
    ImgPrecessPlugin(const void* data, size_t length);
    ImgPrecessPlugin(PreParam pre_param);
    // IPluginV2DynamicExt Funcs
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
        const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;
    // IPluginV2Ext Funcs
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* intputTypes,
        int nbInputs) const noexcept override;
    // IPluginV2 Funcs
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::string name_space_;
    PreParam pre_param_{};
};

class ImgPrecessPluginCreator : public nvinfer1::IPluginCreator {
public:
    ImgPrecessPluginCreator();
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection fc_;
    static std::vector<nvinfer1::PluginField> plugin_attrs_;
    std::string name_space_;
};

}   // ns plugin
}   // ns nvinfer1