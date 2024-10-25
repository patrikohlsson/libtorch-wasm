cmake_minimum_required(VERSION 3.13)
project(InstrumentClassification)

include(FetchContent)

set(LIBTORCH_URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcpu.zip)
set(LIBTORCH_HASH 415c3ed51c766a6ef20dc10b2e60fae7f10a3ae8aa62223d6f4bccc1fc98740b)

# Declare the libtorch content to fetch
FetchContent_Declare(
    libtorch
    URL ${LIBTORCH_URL}
    URL_HASH SHA256=${LIBTORCH_HASH}  # Optional but recommended for integrity verification
)

# Download and make libtorch available
FetchContent_MakeAvailable(libtorch)

# Set the CMAKE_PREFIX_PATH to the extracted libtorch directory
set(CMAKE_PREFIX_PATH ${libtorch_SOURCE_DIR})

# Find the Torch package
find_package(Torch REQUIRED)

# Add your application
add_executable(my_app src/main.cpp)

# Include directories
target_include_directories(my_app
    PRIVATE
        "include"
)

# Link against libtorch
target_link_libraries(my_app
    PRIVATE
        "${TORCH_LIBRARIES}"
)

# Set C++ standard (PyTorch requires C++17 or higher)
set_property(TARGET my_app PROPERTY CXX_STANDARD 17)
