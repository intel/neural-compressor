cmake_minimum_required(VERSION 3.11)
project(kernellibs)
set(HOST_LIBRARY_NAME "kernellibs")

option(SPARSE_LIB_USE_AMX OFF)

set(CMAKE_CXX_FLAGS "-std=c++11 -fpic -Wall -fopenmp -march=native")
if (SPARSE_LIB_USE_AMX)
    set(CMAKE_CXX_FLAGS "-DSPARSE_LIB_USE_AMX ${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
endif()
set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party/")

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

set(WITH_GFLAGS OFF CACHE BOOL "disable gflags for glog")
set(WITH_GTEST OFF CACHE BOOL "disable gtest for glog")
add_subdirectory(${THIRD_PARTY_DIR}/glog)
add_subdirectory(${THIRD_PARTY_DIR}/xbyak)

add_library(${HOST_LIBRARY_NAME} SHARED
    src/interface.cpp
    src/kernel_desc.cpp
    src/kernel.cpp
    src/jit_generator.cpp
    src/kernel_cache.cpp
    src/cpu_engine.cpp
    src/utils.cpp
    src/amx_utils.cpp
    src/kernels/cpu_sparse_matmul_list.cpp
    src/kernels/sparse_data.cpp
    src/kernels/spmm_default.cpp
    src/kernels/spmm_amx_bf16_x16.cpp
    src/jit_domain/jit_spmm_default.cpp
    src/jit_domain/jit_spmm_amx_bf16_x16.cpp
    src/jit_domain/jit_amx_configure.cpp
)

# Header file include path
target_include_directories(${HOST_LIBRARY_NAME}
    PUBLIC
        ./include
)

## link against the third party libraries
target_link_libraries(${HOST_LIBRARY_NAME}
    PUBLIC
        ${CMAKE_THREAD_LIBS_INIT}
        glog
        xbyak
)
