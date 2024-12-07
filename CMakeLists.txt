cmake_minimum_required(VERSION 3.24)
set(PROJECT_NAME yolo11)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -O0 -g")

set(DEVICE_ARCH     "75" CACHE STRING "The device's compute arch to build and run plugin")
set(IS_DYNAMIC_BATCH true CACHE BOOL "The algo's model is of dynamic batch size of not")
set(IS_LOG_PERF      true CACHE BOOL "Whether log model's performance or not")
set(USE_COLOR_LOGGER true CACHE BOOL "Whether use colored logger or not")

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++17,-Wextra,-Wall,-Wno-deprecated-declarations,-O3")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++17 -O0 -g")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
set(CUDA_LIBS ${CUDA_LIBRARIES} nvinfer nvonnxparser nvinfer_plugin cudnn)

find_package(OpenCV 4.0 REQUIRED)
set(OpenCV_LIBS opencv_core opencv_highgui)

link_directories(3rd_libs)

aux_source_directory(src/algos ALGO_SRC_FILES)
aux_source_directory(src/nvprocess NV_SRC_FILES)
####################### trtrun_core.so ########################
add_library(trtrun_core SHARED ${NV_SRC_FILES})
# target_compile_definitions(trtrun_core PUBLIC PRE_NEAREST)
target_include_directories(trtrun_core PUBLIC ${CUDA_INCLUDE_DIRS} src/nvprocess)
target_link_libraries(trtrun_core ${CUDA_LIBS})
set_target_properties(trtrun_core PROPERTIES CUDA_ARCHITECTURES ${DEVICE_ARCH})   # make sure half is supported

######################## target yolo11 ########################
add_executable(${PROJECT_NAME} ${ALGO_SRC_FILES} ./src/main.cpp)
target_compile_definitions(${PROJECT_NAME} PUBLIC PRINT_TIMING=${IS_LOG_PERF} DYNAMIC_BATCH_SIZE=${IS_DYNAMIC_BATCH} USE_COLOR_LOGGER=${USE_COLOR_LOGGER})
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS} src/nvprocess 3rd_libs/include)
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} trtrun_core yaml-cpp)
