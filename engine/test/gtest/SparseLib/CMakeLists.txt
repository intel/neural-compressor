cmake_minimum_required(VERSION 3.11)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -fpic -Wall -fopenmp -march=native")
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

add_subdirectory(${PROJECT_SOURCE_DIR}/../../../SparseLib/third_party/googletest ./googletest)
add_subdirectory(${PROJECT_SOURCE_DIR}/../../../SparseLib/ ./hostlibs)

#### PART2: build gtest cases ####
file(GLOB KERNEL_TEST_CASES_SRC
    test_spmm_default_kernel.cpp
)

function(register_gtest_func exe src)
    add_executable(${exe} ${MAIN_SRC_GTEST} ${src})
    target_link_libraries(${exe}
        PRIVATE
            ${CMAKE_THREAD_LIBS_INIT}
            GTest::gtest
            kernellibs)
endfunction()

foreach(TEST_FILE ${KERNEL_TEST_CASES_SRC})
    get_filename_component(exe_name ${TEST_FILE} NAME_WE)
    register_gtest_func(${exe_name} ${TEST_FILE})
endforeach()
