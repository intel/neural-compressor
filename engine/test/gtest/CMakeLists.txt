cmake_minimum_required(VERSION 3.11)

project(gtest_ut)

set(CMAKE_CXX_FLAGS "-std=c++11 -O0 -fpic -Wall -g -ggdb -march=native")
set(HOST_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../executor/")
set(MAIN_SRC_GTEST "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
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
add_subdirectory(${PROJECT_SOURCE_DIR}/../../executor/third_party/gflags ./gflags)
set(WITH_GFLAGS OFF CACHE BOOL "disable gflags for glog")
add_subdirectory(${PROJECT_SOURCE_DIR}/../../executor/third_party/glog ./glog)
add_subdirectory(${PROJECT_SOURCE_DIR}/../../executor/third_party/oneDNN ./oneDNN)
add_subdirectory(${PROJECT_SOURCE_DIR}/../../executor/third_party/yaml-cpp ./yaml)
add_subdirectory(${PROJECT_SOURCE_DIR}/../../executor/third_party/googletest ./googletest)

#### PART1: build host_operators.so shared library ####
add_library(host_operators SHARED
    ${HOST_SRC_DIR}/src/common.cpp
    ${HOST_SRC_DIR}/src/i_malloc.cpp
    ${HOST_SRC_DIR}/src/operators/binary_add.cpp
    ${HOST_SRC_DIR}/src/operators/layer_norm.cpp
    ${HOST_SRC_DIR}/src/operators/softmax.cpp
    ${HOST_SRC_DIR}/src/operators/padding_sequence.cpp
    ${HOST_SRC_DIR}/src/operators/one_hot.cpp
    ${HOST_SRC_DIR}/src/operators/gather.cpp
    ${HOST_SRC_DIR}/src/operators/strided_slice.cpp
    ${HOST_SRC_DIR}/src/operators/matmul.cpp
    ${HOST_SRC_DIR}/src/operators/inner_product.cpp
    ${HOST_SRC_DIR}/src/operators/reorder.cpp
    ${HOST_SRC_DIR}/src/operators/reduce_mean.cpp
    ${HOST_SRC_DIR}/src/operators/gelu.cpp
    ${HOST_SRC_DIR}/src/operators/position_ids.cpp
    ${HOST_SRC_DIR}/src/operators/token_type_ids.cpp
    ${HOST_SRC_DIR}/src/operators/concat.cpp
    ${HOST_SRC_DIR}/src/operators/embeddingbag.cpp
    ${HOST_SRC_DIR}/src/operators/split.cpp
    ${HOST_SRC_DIR}/src/operators/convolution.cpp
    ${HOST_SRC_DIR}/src/operators/slice.cpp
    ${HOST_SRC_DIR}/src/operators/group_norm.cpp
    ${HOST_SRC_DIR}/src/sparse_operators/sparse_inner_product.cpp
    ${HOST_SRC_DIR}/src/kernels/jit_naive_gemm_kernel.cpp
    ${HOST_SRC_DIR}/src/kernels/jit_naive_matmul.cpp
)

# Header file include path
target_include_directories(host_operators
    PUBLIC
        ${HOST_SRC_DIR}/include/kernels/xbyak
        ${HOST_SRC_DIR}/include/kernels
        ${HOST_SRC_DIR}/include/sparse_operators
        ${HOST_SRC_DIR}/include/operators
        ${HOST_SRC_DIR}/include
        ${HOST_SRC_DIR}/third_party/googletest/googletest/include
        ${HOST_SRC_DIR}/third_party/boost/libs/assert/include
        ${HOST_SRC_DIR}/third_party/boost/libs/core/include
        ${HOST_SRC_DIR}/third_party/boost/libs/move/include
        ${HOST_SRC_DIR}/third_party/boost/libs/config/include
        ${HOST_SRC_DIR}/third_party/boost/libs/container/include
        ${HOST_SRC_DIR}/third_party/boost/libs/intrusive/include
        ${HOST_SRC_DIR}/third_party/boost/libs/static_assert/include
        ${HOST_SRC_DIR}/third_party/boost/libs/type_traits/include
        ${HOST_SRC_DIR}/third_party/boost/libs/interprocess/include
)

# link against the third party libraries
target_link_libraries(host_operators
    PUBLIC
        ${CMAKE_THREAD_LIBS_INIT}
        dnnl
        yaml-cpp
        gflags
        glog
        rt
)


#### PART2: build gtest cases ####
file(GLOB OPS_TEST_CASES_SRC
    test_i_malloc.cpp
    test_binary_add_op.cpp
    test_layer_norm_op.cpp
    test_softmax_op.cpp
    test_padding_sequence_op.cpp
    test_one_hot_op.cpp
    test_gather_op.cpp
    test_strided_slice_op.cpp
    test_matmul_op.cpp
    test_inner_product_op.cpp
    test_reorder_op.cpp
    test_reduce_mean_op.cpp
    test_gelu_op.cpp
    test_position_ids_op.cpp
    test_token_type_ids_op.cpp
    test_concat_op.cpp
    test_split_op.cpp
    test_slice_op.cpp
    test_embeddingbag_op.cpp
    test_convolution_op.cpp
    test_group_norm_op.cpp
    kernels/test_naive_gemm.cpp
)

function(register_gtest_func exe src)
    add_executable(${exe} ${MAIN_SRC_GTEST} ${src})
    target_link_libraries(${exe}
        PRIVATE
            ${CMAKE_THREAD_LIBS_INIT}
            GTest::gtest
            host_operators)
endfunction()

foreach(TEST_FILE ${OPS_TEST_CASES_SRC})
    get_filename_component(exe_name ${TEST_FILE} NAME_WE)
    register_gtest_func(${exe_name} ${TEST_FILE})
endforeach()
