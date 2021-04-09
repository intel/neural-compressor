cmake_minimum_required(VERSION 3.1)

project(mlperf_loadgen)

# The mlperf_loadgen version.
set(mlperf_loadgen_VERSION_MAJOR 0)
set(mlperf_loadgen_VERSION_MINOR 5)
message("mlperf_loadgen v${mlperf_loadgen_VERSION_MAJOR}.${mlperf_loadgen_VERSION_MINOR}")

# Set build options. NB: CXX_STANDARD is supported since CMake 3.1.
if (NOT MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -W -Wall")
endif()
message(STATUS "Using C++ compiler flags: ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD "14")
message(STATUS "Using C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Using static linker flags: ${CMAKE_STATIC_LINKER_FLAGS}")
message(STATUS "Using shared linker flags: ${CMAKE_SHARED_LINKER_FLAGS}")

# Output directory for libraries.
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
message(STATUS "Using output path: ${LIBRARY_OUTPUT_PATH}")

# Detect Python to use for generating source file with version info.
# NB: PythonInterp has been deprecated since CMake 3.12
# but it works with earlier versions of CMake.
find_package(PythonInterp)
message(STATUS "Using Python interpreter: ${PYTHON_EXECUTABLE}")

# Generate source file with version info.
execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/version_generator.py ${CMAKE_BINARY_DIR}/version_generated.cc ${CMAKE_CURRENT_SOURCE_DIR})

# Add source files.
set(SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings/c_api.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings/c_api.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/issue_query_controller.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/loadgen.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/logging.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/logging.h
  ${CMAKE_CURRENT_SOURCE_DIR}/test_settings_internal.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/test_settings_internal.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/version.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/version.h
  ${CMAKE_BINARY_DIR}/version_generated.cc
)

add_library(mlperf_loadgen STATIC ${SOURCE})
target_link_libraries(mlperf_loadgen)

# Install library and headers.
install(TARGETS mlperf_loadgen
	DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
	DESTINATION ${CMAKE_INSTALL_PREFIX}/include FILES_MATCHING PATTERN "*.h")
