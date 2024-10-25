FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

ENV BASE_DIR=/home/ubuntu
RUN mkdir -p ${BASE_DIR}

WORKDIR ${BASE_DIR}

# Update the package list
RUN apt update -y && apt upgrade -y

# Install dependencies
RUN apt install -y build-essential cmake python3 python3-pip python3-yaml git wget
RUN python3 -m pip install typing-extensions

# Copy files
COPY native/CMakeLists.txt ${BASE_DIR}
COPY src/ ${BASE_DIR}/src/
COPY include/ ${BASE_DIR}/include/

# Configure and build
RUN cmake -Bbuild
RUN cmake --build build --config Release -j16 --target my_app

# Copy the pretrained model
COPY resnet50.pt ${BASE_DIR}/build/resnet50.pt

# Run the application
CMD cd build && ./my_app