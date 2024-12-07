#pragma once
#include <stdio.h>
#include <nvtx3/nvToolsExt.h>
#define REPEATS_NUM 10

#define CHECK_CUDA_FUNC(call) {                         \
    const cudaError_t res = call;                       \
    if (res != cudaSuccess) {                           \
        printf("Error at: %s:%d, ", __FILE__, __LINE__);\
        printf("Error code:%d, reason: %s\n", res, cudaGetErrorString(res));\
        exit(1);                                        \
    }                                                   \
}

#define CHECK_KERNEL_TIME(kernel_call) {\
    float time_sum = 0;                 \
    float time2_sum = 0;                \
    for (int repeat = 0; repeat <= REPEATS_NUM; repeat++) {  \
        float delta_time;                                   \
        cudaEvent_t beg, end;                               \
        CHECK_CUDA_FUNC(cudaEventCreate(&beg));             \
        CHECK_CUDA_FUNC(cudaEventCreate(&end));             \
        CHECK_CUDA_FUNC(cudaEventRecord(beg));              \
        cudaEventQuery(beg);                \
        kernel_call;                        \
        CHECK_CUDA_FUNC(cudaEventRecord(end));              \
        CHECK_CUDA_FUNC(cudaEventSynchronize(end));         \
        CHECK_CUDA_FUNC(cudaEventElapsedTime(&delta_time, beg, end));\
        printf("%dth GPU Time = %g ms.\n", repeat, delta_time);\
        CHECK_CUDA_FUNC(cudaEventDestroy(beg));                \
        CHECK_CUDA_FUNC(cudaEventDestroy(end));                \
        if (repeat > 0) {                   \
            time_sum += delta_time;                           \
            time2_sum += delta_time * delta_time;             \
        }                                   \
    }                                       \
    const float time_avg = time_sum / REPEATS_NUM;            \
    const float time_std = sqrt(time2_sum / REPEATS_NUM - time_avg * time_avg); \
    printf("Avg GPU Time = %g +- %g ms .\n", time_avg, time_std);               \
}

#define ADD_PROFILE_MARK(kerne_call, name) {  \
    nvtxRangePush(name);        \
    CHECK_CUDA_FUNC(kerne_call);        \
    nvtxRangePop();                     \
}
