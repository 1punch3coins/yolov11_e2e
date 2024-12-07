#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include "img_precess_k.hpp"
#include <iostream>

// src_crop--crop+rescale-->dst_crop, hwc--permute-->chw, bgr--shuffle-->rgb, uint8 or float--normalize--->float
template <typename T>
static void __global__ crop_scale_permute_shuffle_normalize(const T* src, float* channel_r, float* channel_g, float* channel_b, const PreParam param) {
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i > param.dst_crop.w || j > param.dst_crop.h) {return;}
    unsigned dst_ind = j*param.dst_step+i;
    float3 rgb;
    
    float scaled_i = param.scale_inv.x * i + param.src_crop.l;
    float scaled_j = param.scale_inv.y * j + param.src_crop.t;
    float f_scaled_i = floorf(scaled_i);
    float f_scaled_j = floorf(scaled_j);
    unsigned src_i = scaled_i - f_scaled_i < 0.5 ? f_scaled_i : f_scaled_i+1;
    unsigned src_j = scaled_j - f_scaled_j < 0.5 ? f_scaled_j : f_scaled_j+1;
    src_i = src_i >= param.src_size.x ? param.src_size.x-1 : src_i;
    src_j = src_j >= param.src_size.y ? param.src_size.y-1 : src_j;
    unsigned src_ind = (src_j*param.src_step+src_i)*3;
    rgb.x = (src[src_ind+2]*1.0f - param.mean.x)*param.norm_inv.x;
    rgb.y = (src[src_ind+1]*1.0f - param.mean.y)*param.norm_inv.y;
    rgb.z = (src[src_ind+0]*1.0f - param.mean.z)*param.norm_inv.z;
    channel_r[dst_ind] = rgb.x;
    channel_g[dst_ind] = rgb.y;
    channel_b[dst_ind] = rgb.z;
}

// src_crop--crop+rescale-->dst_crop, hwc--permute-->chw, bgr--shuffle-->rgb, half--normalize--->half
static void __global__ crop_scale_permute_shuffle_normalize(const half* src, half* channel_r, half* channel_g, half* channel_b, const PreParam param) {
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i > param.dst_crop.w || j > param.dst_crop.h) {return;}
    unsigned dst_ind = j*param.dst_step+i;
    
    float scaled_i = param.scale_inv.x * i + param.src_crop.l;
    float scaled_j = param.scale_inv.y * j + param.src_crop.t;
    float f_scaled_i = floorf(scaled_i);
    float f_scaled_j = floorf(scaled_j);
    unsigned src_i = scaled_i - f_scaled_i < 0.5 ? f_scaled_i : f_scaled_i+1;
    unsigned src_j = scaled_j - f_scaled_j < 0.5 ? f_scaled_j : f_scaled_j+1;
    src_i = src_i >= param.src_size.x ? param.src_size.x-1 : src_i;
    src_j = src_j >= param.src_size.y ? param.src_size.y-1 : src_j;
    unsigned src_ind = (src_j*param.src_step+src_i)*3;
    channel_r[dst_ind] = __hmul(__hsub(src[src_ind+2], param.mean16.x), param.norm_inv16.x);
    channel_g[dst_ind] = __hmul(__hsub(src[src_ind+1], param.mean16.y), param.norm_inv16.y);
    channel_b[dst_ind] = __hmul(__hsub(src[src_ind+0], param.mean16.z), param.norm_inv16.z);
}

template <typename T0, typename T1>
cudaError_t launch_pre_kernel(const T0* d_mat_src, T1* d_mat_dst, const PreParam& param, void* stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    dim3 threads(16,16);
    dim3 blocks((param.dst_crop.w+threads.x-1)/threads.x, (param.dst_crop.h+threads.y-1)/threads.y);    // a thread for a output pixel
    unsigned int planar_size = param.dst_size.x*param.dst_size.y;
    unsigned int lt_loc = param.dst_crop.t * param.dst_step + param.dst_crop.l;
    crop_scale_permute_shuffle_normalize<<<blocks,threads,0,stream>>>(d_mat_src, d_mat_dst+lt_loc, d_mat_dst+planar_size+lt_loc, d_mat_dst+planar_size*2+lt_loc, param);
    cudaError_t err = cudaGetLastError();
    return err;
}
template cudaError_t launch_pre_kernel<uint8_t, float>(const uint8_t* d_mat_src, float* d_mat_dst, const PreParam& param, void* stream_ptr);
template cudaError_t launch_pre_kernel<float, float>(const float* d_mat_src, float* d_mat_dst, const PreParam& param, void* stream_ptr);
template cudaError_t launch_pre_kernel<half, half>(const half* d_mat_src, half* d_mat_dst, const PreParam& param, void* stream_ptr);

// multi_input--pre_kernel-->multi_output
template <typename T0, typename T1>
cudaError_t launch_batched_pre_kernel(const std::vector<T0*>& d_mat_src_vec, T1* d_mat_dst, const PreParam& param, void* stream_ptr) {
    unsigned int dst_mat_size = param.dst_size.y*param.dst_step*3*sizeof(T1);
    for (const auto& d_mat_src: d_mat_src_vec) {
        launch_pre_kernel(d_mat_src, d_mat_dst, param, stream_ptr);
        d_mat_dst += dst_mat_size;
    }
    cudaError_t err = cudaGetLastError();
    return err;
}
template cudaError_t launch_batched_pre_kernel<uint8_t, float>(const std::vector<uint8_t*>& d_mat_src_vec, float* d_mat_dst, const PreParam& param, void* stream_ptr);
template cudaError_t launch_batched_pre_kernel<float, float>(const std::vector<float*>& d_mat_src_vec, float* d_mat_dst, const PreParam& param, void* stream_ptr);

// multi_input--pre_kernel-->multi_output
template <typename T0, typename T1>
cudaError_t launch_batched_pre_kernel(const T0* d_mat_src, T1* d_mat_dst, const PreParam& param, void* stream_ptr) {
    // unsigned int src_mat_size = param.src_size.y*param.src_step*3*sizeof(T0);
    // unsigned int dst_mat_size = param.dst_size.y*param.dst_step*3*sizeof(T1); // cause error, just a reminder
    unsigned int src_mat_size = param.src_size.y*param.src_step*3;
    unsigned int dst_mat_size = param.dst_size.y*param.dst_step*3;
    for (unsigned i = 0; i < param.batch_size; i++) {
        launch_pre_kernel(d_mat_src, d_mat_dst, param, stream_ptr);
        d_mat_src += src_mat_size;
        d_mat_dst += dst_mat_size;
    }
    cudaError_t err = cudaGetLastError();
    return err;
}
template cudaError_t launch_batched_pre_kernel<uint8_t, float>(const uint8_t* d_mat_src, float* d_mat_dst, const PreParam& param, void* stream_ptr);
template cudaError_t launch_batched_pre_kernel<float, float>(const float* d_mat_src, float* d_mat_dst, const PreParam& param, void* stream_ptr);
template cudaError_t launch_batched_pre_kernel<half, half>(const half* d_mat_src, half* d_mat_dst, const PreParam& param, void* stream_ptr);
