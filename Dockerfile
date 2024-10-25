FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

ENV BASE_DIR=/home/ubuntu
ARG JOBS=4
RUN mkdir -p ${BASE_DIR}

WORKDIR ${BASE_DIR}

# Update the package list
RUN apt update -y && apt upgrade -y

# Install dependencies
RUN apt install -y build-essential cmake python3 python3-pip python3-yaml libopenblas-dev git wget
RUN python3 -m pip install typing-extensions

# Install Emscripten
RUN git clone https://github.com/emscripten-core/emsdk.git
RUN cd emsdk && ./emsdk install latest && ./emsdk activate latest
ENV PATH="${BASE_DIR}/emsdk:${BASE_DIR}/emsdk/node/18.20.3_64bit/bin:${BASE_DIR}/emsdk/upstream/emscripten:${PATH}"

# Create the source directory and a placeholder source file
RUN mkdir ${BASE_DIR}/src && touch ${BASE_DIR}/src/main.cpp

COPY CMakeLists.txt ${BASE_DIR}

# We manually build protoc on the host architecture to ensure we have a version compatible with PyTorch
RUN cmake -Bbuild -DBUILD_CUSTOM_PROTOBUF=ON -DPROTOBUF_PROTOC_EXECUTABLE="" -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=""
RUN cmake --build build --config Release -j${JOBS} --target protoc
RUN ls ./build/bin/*
RUN cp ./build/bin/protoc* /usr/local/bin

# Clean up the build directory
RUN rm -rf build 

COPY pytorch_emscripten_fix.patch ${BASE_DIR}

# Prebuild dependencies
RUN emcmake cmake -Bbuild
RUN cmake --build build --config Release -j${JOBS} --target torch_cpu c10

COPY src/ ${BASE_DIR}/src/
COPY include/ ${BASE_DIR}/include/
COPY resnet50.pt ${BASE_DIR}/build/resnet50.pt
RUN ls ${BASE_DIR}/src ${BASE_DIR}/build

RUN cmake --build build --config Release -j${JOBS} --target my_app

# Start a web server
CMD cp build/my_app.html build/index.html && python3 -m http.server 8080 --directory build
