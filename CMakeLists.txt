cmake_minimum_required(VERSION 3.13)
project(BuildLibtorch)

include(FetchContent)

# Most options are turned off for Emscripten
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(BUILD_PYTHON OFF CACHE BOOL "")
set(BUILD_TEST OFF CACHE BOOL "")
set(BUILD_BINARY OFF CACHE BOOL "")
set(BUILD_CUSTOM_PROTOBUF OFF CACHE BOOL "")
set(PROTOBUF_PROTOC_EXECUTABLE "/usr/local/bin/protoc")
set(CAFFE2_CUSTOM_PROTOC_EXECUTABLE "/usr/local/bin/protoc")
set(CAFFE2_USE_MSVC_STATIC_RUNTIME OFF CACHE BOOL "")
set(USE_CUDA OFF CACHE BOOL "")  # Set to ON if CUDA is needed
set(USE_CUDNN OFF CACHE BOOL "")
set(USE_BLAS ON CACHE BOOL "")
set(USE_MKLDNN OFF CACHE BOOL "")
set(USE_NNPACK OFF CACHE BOOL "")
set(USE_NUMPY OFF CACHE BOOL "")
set(USE_QNNPACK OFF CACHE BOOL "")
set(USE_XNNPACK OFF CACHE BOOL "")
set(USE_OPENMP OFF CACHE BOOL "")
set(USE_DISTRIBUTED OFF CACHE BOOL "")
set(USE_FUNCTORCH OFF CACHE BOOL "")
set(USE_FBGEMM OFF CACHE BOOL "")
set(USE_KINETO OFF CACHE BOOL "")
set(USE_ONNX OFF CACHE BOOL "")
set(USE_FLATBUFFERS OFF CACHE BOOL "")
set(USE_FUNCTORCH OFF CACHE BOOL "")
set(USE_FBGEMM OFF CACHE BOOL "")
set(USE_KINETO OFF CACHE BOOL "")
set(USE_ONNX OFF CACHE BOOL "")
set(USE_FLATBUFFERS OFF CACHE BOOL "")

# Set directories for source, build, and installation
set(PYTORCH_VERSION v2.5.0)

# Find Python 3
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python 3.10 COMPONENTS Interpreter REQUIRED)

# Download libtorch
FetchContent_Declare(
    Torch
    GIT_REPOSITORY https://github.com/pytorch/pytorch.git
    GIT_TAG ${PYTORCH_VERSION}
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(Torch)

# Add torch to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${PYTORCH_INSTALL_DIR}/share/cmake/Torch")

# Patch cpuinfo
add_custom_target(patch_cpuinfo
    COMMAND sed -i "s/|Android|FreeBSD)/|Android|FreeBSD|Emscripten)/g" third_party/cpuinfo/CMakeLists.txt
    BYPRODUCTS ${torch_SOURCE_DIR}/third_party/cpuinfo/CMakeLists.txt
    DEPENDS ${torch_SOURCE_DIR}/third_party/cpuinfo/CMakeLists.txt
    WORKING_DIRECTORY ${torch_SOURCE_DIR}
    VERBATIM
)
add_dependencies(cpuinfo patch_cpuinfo)

# Patch PyTorch
add_custom_target(patch_pytorch
    COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/pytorch_emscripten_fix.patch || true
    BYPRODUCTS ${torch_SOURCE_DIR}/c10/macros/Macros.h
    DEPENDS ${torch_SOURCE_DIR}/c10/macros/Macros.h
    WORKING_DIRECTORY ${torch_SOURCE_DIR}
    VERBATIM
)
add_dependencies(torch_cpu patch_pytorch)

# Add application
add_executable(my_app src/main.cpp)

message(STATUS "torch_SOURCE_DIR ${torch_SOURCE_DIR}")

# Include directories
target_include_directories(my_app
    PRIVATE
        "include"
        "${torch_SOURCE_DIR}"
        "${torch_SOURCE_DIR}/aten/src"
        "${torch_SOURCE_DIR}/torch/csrc/api/include"
        "${CMAKE_CURRENT_BINARY_DIR}/aten/src"
        "${CMAKE_CURRENT_BINARY_DIR}"
)

# Ensure the app compiles after libtorch
add_dependencies(my_app torch_cpu c10)

set_target_properties(my_app
    PROPERTIES
        OUTPUT_NAME "my_app"
        SUFFIX ".html")

target_link_options(my_app
    PRIVATE
        # Model needs to be embedded in the emscripten filesystem
        -sEXPORTED_FUNCTIONS=_main,_load_model,_run_model
        --preload-file ${CMAKE_CURRENT_BINARY_DIR}/resnet50.pt@resnet50.pt
        -sALLOW_MEMORY_GROWTH=1
)

set(TORCH_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib")

# Link against libtorch
target_link_libraries(my_app
    PRIVATE
        -Wl,--whole-archive # Seems to be needed, see PyTorch CMakeLists.txt, but slow as hell
        ${TORCH_LIB_DIR}/libtorch_cpu.a
        -Wl,--no-whole-archive
        ${TORCH_LIB_DIR}/libc10.a
        ${TORCH_LIB_DIR}/libcpuinfo.a
)

# Set C++ standard (PyTorch requires C++17 or higher)
set_property(TARGET my_app PROPERTY CXX_STANDARD 17)
