project(engine)
if(CMAKE_VERSION VERSION_LESS 3.12)
    message("${CMAKE_VERSION} is less then the minimum required for engine build which is cmake 3.12, skip the engine build")
elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.2.1)
    message("GCC version less then the minimum required for engine build which is 7.2.1, skip the engine build")
else()
cmake_minimum_required(VERSION 3.12)


set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -lm -ftree-loop-vectorize -fpic -Wall -fopenmp -march=native -Wl,-rpath,'$ORIGIN'")
set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Os -Wall -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Os -Wall -s")
#execute_process(COMMAND cat /proc/cpuinfo
#                COMMAND head -n 26
#                COMMAND grep -c avx512
#                OUTPUT_VARIABLE AVX_RESULT)
#if(NOT AVX_RESULT EQUAL "1")
#  message(FATAL_ERROR "AVX512 is necessary for engine, please check your device")
#endif()

# -lpthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)
if(Threads_FOUND)
    message("Threads found: ${CMAKE_THREAD_LIBS_INIT}")
else()
    message(STATUS "Cannot find Threads")
endif()

# -fopenmp
find_package(OpenMP)
if(OpenMP_FOUND)
    message("OpenMP found: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(${CMAKE_CXX_FLAGS})
else()
    message(STATUS "Cannot find OpenMP")
endif()

set(GFLAGS_USE_TARGET_NAMESPACE TRUE)
add_subdirectory(${THIRD_PARTY_DIR}/gflags)
set(WITH_GFLAGS OFF CACHE BOOL "disable gflags for glog")
add_subdirectory(${THIRD_PARTY_DIR}/glog)
add_subdirectory(${THIRD_PARTY_DIR}/oneDNN)
add_subdirectory(${THIRD_PARTY_DIR}/pybind11)
add_subdirectory(${THIRD_PARTY_DIR}/yaml-cpp)

add_library(engine SHARED
    src/model.cpp
    src/common.cpp
    src/i_malloc.cpp
    src/operators/input.cpp
    src/operators/output.cpp
    src/operators/binary_add.cpp
    src/operators/inner_product.cpp
    src/operators/layer_norm.cpp
    src/operators/matmul.cpp
    src/operators/one_hot.cpp
    src/operators/padding_sequence.cpp
    src/operators/reorder.cpp
    src/operators/softmax.cpp
    src/operators/reshape.cpp
    src/operators/gather.cpp
    src/operators/strided_slice.cpp
    src/operators/quantize.cpp
    src/operators/reduce_mean.cpp
    # src/operators/transpose.cpp
    src/operators/gelu.cpp
    src/operators/position_ids.cpp
    src/operators/token_type_ids.cpp
    src/operators/concat.cpp
    src/operators/embeddingbag.cpp
    src/operators/split.cpp
    src/operators/convolution.cpp
    src/operators/merged_embeddingbag.cpp
    src/operators/slice.cpp
    src/operators/group_norm.cpp
    src/sparse_operators/sparse_inner_product.cpp
    src/kernels/jit_naive_gemm_kernel.cpp
    src/kernels/jit_naive_matmul.cpp
)

# Header file include path
target_include_directories(engine
    PUBLIC
        ./include/kernels
        ./include/kernels/xbyak
        ./include/sparse_operators
        ./include/operators
        ./include
)

## link against the third party libraries
target_link_libraries(engine
    PUBLIC
        ${CMAKE_THREAD_LIBS_INIT}
        dnnl
        yaml-cpp
        gflags
        glog
)

## pybind11
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.9)
  cmake_policy(SET CMP0069 NEW)
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
pybind11_add_module(engine_py ${PROJECT_SOURCE_DIR}/python/bind_engine.cpp)
target_include_directories(engine_py
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/python
)

# link against the third party libraries
target_link_libraries(engine_py
    PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
        engine
)

# build inferencer executable
add_executable(inferencer
    src/inferencer.cc
)

target_include_directories(inferencer
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(inferencer
    PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
        engine
)

endif()
