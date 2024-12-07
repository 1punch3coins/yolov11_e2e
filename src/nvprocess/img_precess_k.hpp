#pragma once
#include <vector>
#include "img_precess_p.hpp"

using PreParam = nvinfer1::plugin::PreParam;
using Crop = nvinfer1::plugin::Crop;

template<typename T0, typename T1>
cudaError_t launch_pre_kernel(const T0* d_mat_src, T1* d_mat_dst, const PreParam& param, void* stream_ptr);

template<typename T0, typename T1>
cudaError_t launch_batched_pre_kernel(const std::vector<T0*>& d_mat_src_vec, T1* d_mat_dst, const PreParam& param, void* stream_ptr);
template<typename T0, typename T1>
cudaError_t launch_batched_pre_kernel(const T0* d_mat_src, T1* d_mat_dst, const PreParam& param, void* stream_ptr);
