#include "img_precess_p.hpp"
#include "img_precess_k.hpp"
#include <iostream>
#include <cassert>

using namespace nvinfer1;
using nvinfer1::plugin::ImgPrecessPlugin;
using nvinfer1::plugin::ImgPrecessPluginCreator;
static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"ImgPrecessPlugin"};

// initialize static class fields
PluginFieldCollection ImgPrecessPluginCreator::fc_{};
std::vector<PluginField> ImgPrecessPluginCreator::plugin_attrs_;

template<typename T>
static void writeToBuffer(char*& buffer, const T& val) {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);    // offset pointer to the next
}

template<typename T>
static T readFromBuffer(const char*& buffer) {
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);    // offset pointer to the next
    return val;
}

ImgPrecessPlugin::ImgPrecessPlugin(PreParam pre_param):    // constructor used for parser
    pre_param_(std::move(pre_param))
{

}

ImgPrecessPlugin::ImgPrecessPlugin(const void* data, size_t length) {     // constructor used in deserialization
    const char* d = reinterpret_cast<const char*>(data);
    pre_param_ = readFromBuffer<PreParam>(d);
}

nvinfer1::IPluginV2DynamicExt* ImgPrecessPlugin::clone() const noexcept {    // deep clone func used by trt's builder, network or engine
    auto* plugin = new ImgPrecessPlugin(pre_param_);
    plugin->setPluginNamespace(name_space_.c_str());
    return plugin;
}

bool ImgPrecessPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept { // func to check out plugin's supported input and output types
    // inputs are inOut[0,(nbInputs-1)] and outputs are inOut[nbInputs,(nbInputs+nbOutputs-1)]
    assert(nbInputs == 1);  // the plugin has 1 input
    assert(nbOutputs == 1); // the plugin has 1 output
    const PluginTensorDesc& in = inOut[pos];
    const PluginTensorDesc& out = inOut[nbInputs];
    switch(pos) {
        case 0: return ((in.type == nvinfer1::DataType::kFLOAT || in.type == nvinfer1::DataType::kHALF) && (in.format == TensorFormat::kLINEAR));
        case 1: return (inOut[0].type == nvinfer1::DataType::kFLOAT && out.type == nvinfer1::DataType::kFLOAT && out.format == TensorFormat::kLINEAR)
                    || (inOut[0].type == nvinfer1::DataType::kHALF && out.type == nvinfer1::DataType::kHALF && out.format == TensorFormat::kLINEAR);
    }
    return false;
}

nvinfer1::DimsExprs ImgPrecessPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, 
    int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept { // func to get plugin's output's dimension
    assert(outputIndex == 0);    // the plugin's output has only one ouptut at index 0
    nvinfer1::DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];           // batch_size set the same as input's
    output.d[1] = inputs[0].d[3];           // output's C is the same as input's C
    output.d[2] = exprBuilder.constant(pre_param_.dst_size.y);  // H
    output.d[3] = exprBuilder.constant(pre_param_.dst_size.x);  // W
    return output;
}

nvinfer1::DataType ImgPrecessPlugin::getOutputDataType( int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {   // func to get plugin's output's data type
    return inputTypes[0];
}

int ImgPrecessPlugin::getNbOutputs() const noexcept {// func to get num of plugin's output
    return 1;
}

void ImgPrecessPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept { // nothing to do
    // this member function will only be called during engine build time
    return;
}

size_t ImgPrecessPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbINputs, const nvinfer1::PluginTensorDesc* ouputs, int nbOutputs) const noexcept {
    // no scratch space is need for this plugin
    return 0;
}

const char* ImgPrecessPlugin::getPluginType() const noexcept {
    return PLUGIN_NAME;
}

const char* ImgPrecessPlugin::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

int ImgPrecessPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, 
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept { // core func to setup calculation, here link to a cuda kernel launch func
    try {
        unsigned batch_size = inputDesc[0].dims.d[0];
        pre_param_.batch_size = batch_size;

        int status = -1;
        nvinfer1::DataType inputType = inputDesc[0].type;
        nvinfer1::DataType outputType = outputDesc[0].type;
        if (inputType == nvinfer1::DataType::kHALF) {
            auto raw_img_data = static_cast<const half*>(inputs[0]);
            auto precessed_data = static_cast<half*>(outputs[0]);
            status = launch_batched_pre_kernel(
                raw_img_data, precessed_data, pre_param_, stream
            );
            assert(status == 0);
            return status;
        } else if (inputType == nvinfer1::DataType::kFLOAT) {
            auto raw_img_data = static_cast<const float*>(inputs[0]);
            auto precessed_data = static_cast<float*>(outputs[0]);
            status = launch_batched_pre_kernel(
                raw_img_data, precessed_data, pre_param_, stream
            );
            assert(status == 0);
            return status;
        } else {
            assert(status == 0);
            return status;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return -1;
}

int ImgPrecessPlugin::initialize() noexcept {
    // The configuration is known at this time, and the inference engine is being created, so the plugin can set up its internal data structures and
    // prepare for execution. Such setup might include initializing libraries, allocating memory, etc. In our case, we don't need to prepare anything.
    return 0;
}

void ImgPrecessPlugin::terminate() noexcept {
}

size_t ImgPrecessPlugin::getSerializationSize() const noexcept {
    return sizeof(PreParam);
}

void ImgPrecessPlugin::serialize(void* buffer) const noexcept {  // serialize the attrs from onnx node
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<PreParam>(d, pre_param_);
}

void ImgPrecessPlugin::destroy() noexcept {
    delete this;
}

void ImgPrecessPlugin::setPluginNamespace(const char* libNamespace) noexcept {
    name_space_ = libNamespace;
}

const char* ImgPrecessPlugin::getPluginNamespace() const noexcept {
    return name_space_.c_str();
}


ImgPrecessPluginCreator::ImgPrecessPluginCreator() {
    // declare the onnx attrs that the onnx parser will collect from the onnx model that contains the PPScatter node.
    plugin_attrs_.clear();
    plugin_attrs_.emplace_back(PluginField("src_crop", nullptr, PluginFieldType::kINT32, 4));
    plugin_attrs_.emplace_back(PluginField("dst_crop", nullptr, PluginFieldType::kINT32, 4));
    plugin_attrs_.emplace_back(PluginField("src_size", nullptr, PluginFieldType::kINT32, 2));
    plugin_attrs_.emplace_back(PluginField("dst_size", nullptr, PluginFieldType::kINT32, 2));
    plugin_attrs_.emplace_back(PluginField("scale_inv", nullptr, PluginFieldType::kFLOAT32, 2));
    plugin_attrs_.emplace_back(PluginField("mean", nullptr, PluginFieldType::kFLOAT32, 3));
    plugin_attrs_.emplace_back(PluginField("norm_inv", nullptr, PluginFieldType::kFLOAT32, 3));
    plugin_attrs_.emplace_back(PluginField("src_step", nullptr, PluginFieldType::kINT32, 1));
    plugin_attrs_.emplace_back(PluginField("dst_step", nullptr, PluginFieldType::kINT32, 1));
    fc_.nbFields = plugin_attrs_.size();
    fc_.fields = plugin_attrs_.data();
    for (int i = 0; i < fc_.nbFields; i++) {
        printf("%s %d %d\n", fc_.fields[0].name, fc_.fields[0].type, fc_.fields[0].length);
    }
}

const char* ImgPrecessPluginCreator::getPluginName() const noexcept {
    return PLUGIN_NAME;
}

const char* ImgPrecessPluginCreator::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

const PluginFieldCollection* ImgPrecessPluginCreator::getFieldNames() noexcept {
    for (int i = 0; i < fc_.nbFields; i++) {
        printf("%s %d %d\n", fc_.fields[i].name, fc_.fields[i].type, fc_.fields[i].length);
    }
    return &fc_;
}

void ImgPrecessPluginCreator::setPluginNamespace(const char* libNamespace) noexcept {
    name_space_ = libNamespace;
}

const char* ImgPrecessPluginCreator::getPluginNamespace() const noexcept {
    return name_space_.c_str();
}

// used in model building phase
IPluginV2* ImgPrecessPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
     // 1. the attributes from the ONNX node will be parsed and passed via the second parameter
    const PluginField* fields = fc->fields;
    int nbFileds = fc->nbFields;    // number of attributes
    PreParam pre_param;
    for (int i = 0; i < nbFileds; i++) {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "src_crop")) {
            const unsigned* src_crop = static_cast<const unsigned*>(fields[i].data);
            pre_param.src_crop.l = src_crop[0];
            pre_param.src_crop.t = src_crop[1];
            pre_param.src_crop.w = src_crop[2];
            pre_param.src_crop.h = src_crop[3];
        }
        if (!strcmp(attr_name, "dst_crop")) {
            const unsigned* dst_crop = static_cast<const unsigned*>(fields[i].data);
            pre_param.dst_crop.l = dst_crop[0];
            pre_param.dst_crop.t = dst_crop[1];
            pre_param.dst_crop.w = dst_crop[2];
            pre_param.dst_crop.h = dst_crop[3];
        }
        if (!strcmp(attr_name, "src_size")) {
            const unsigned* src_size = static_cast<const unsigned*>(fields[i].data);
            pre_param.src_size.y = src_size[0];
            pre_param.src_size.x = src_size[1];
        }
        if (!strcmp(attr_name, "dst_size")) {
            const unsigned* dst_size = static_cast<const unsigned*>(fields[i].data);
            pre_param.dst_size.y = dst_size[0];
            pre_param.dst_size.x = dst_size[1];
        }
        if (!strcmp(attr_name, "src_step")) {
            const unsigned* src_step = static_cast<const unsigned*>(fields[i].data);
            pre_param.src_step = src_step[0];
        }
        if (!strcmp(attr_name, "dst_step")) {
            const unsigned* dst_step = static_cast<const unsigned*>(fields[i].data);
            pre_param.dst_step = dst_step[0];
        }
        if (!strcmp(attr_name, "scale_inv")) {
            const float* scales = static_cast<const float*>(fields[i].data);
            pre_param.scale_inv.y = scales[0];
            pre_param.scale_inv.x = scales[1];
        }
        if (!strcmp(attr_name, "mean")) {
            const float* means = static_cast<const float*>(fields[i].data);
            pre_param.mean.x = means[0];
            pre_param.mean.y = means[1];
            pre_param.mean.z = means[2];
        }
        if (!strcmp(attr_name, "norm_inv")) {
            const float* norms = static_cast<const float*>(fields[i].data);
            pre_param.norm_inv.x = norms[0];
            pre_param.norm_inv.y = norms[1];
            pre_param.norm_inv.z = norms[2];
        }
    }
    pre_param.mean16.x = __float2half(pre_param.mean.x);
    pre_param.mean16.y = __float2half(pre_param.mean.y);
    pre_param.mean16.z = __float2half(pre_param.mean.z);
    pre_param.norm_inv16.x = __float2half(pre_param.norm_inv.x);
    pre_param.norm_inv16.y = __float2half(pre_param.norm_inv.y);
    pre_param.norm_inv16.z = __float2half(pre_param.norm_inv.z);
    // 2. use the parsed attributes to create a plugin
    auto* plugin = new ImgPrecessPlugin(pre_param);
    return plugin;
}

// used in model deserialization phase
IPluginV2* ImgPrecessPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
    auto* plugin = new ImgPrecessPlugin(serialData, serialLength);
    return plugin;
}

REGISTER_TENSORRT_PLUGIN(ImgPrecessPluginCreator);