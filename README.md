# Libtorch WebAssembly

This project compiles libtorch to WebAssembly using Emscripten.
Includes an example that loads a pre-trained ResNet50 model that classifies images in the browser.

## Requirements

- [PyTorch](https://pytorch.org/get-started/locally/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Docker](https://www.docker.com/get-started)

### Setup

Exporting the example model requires PyTorch and Torchvision. Setting up a new conda environment is recommended:

```bash
conda create -n libtorch-wasm python=3.11
conda activate libtorch-wasm
conda install pytorch torchvision cpuonly -c pytorch
```

## Exporting the model

To export the pre-trained example ResNet50 model, run the following command in the root of the project directory:

```bash
python export-model.py
```

This will generate a resnet50.pt file in the same directory.

## Compiling the WASM build
The project uses Docker. The primary WebAssembly build is located in the root directory. To compile it, run the following command:

```bash
docker build -t resnet-torch-wasm .
```
*Note: This is slow.*

## Running the WASM build

Once the WebAssembly image is built, you can start a server to host the example page:

```bash
docker run -it --rm -p 8080:8080 resnet-torch-wasm
```

After running this command, the server will be accessible at http://localhost:8080, where you can see the example page in action.

## Code testing

For testing purposes you may also compile a Ubuntu native binary. This will compile an executable linked against a pre-built `libtorch` which is quick and practical for prototyping. To compile the native build, run the following command:

```bash
docker build -f native/Dockerfile -t resnet-torch-native .
```

## Running test builds

The native image will just run the example code and exit. Use the following command:

```bash
docker run -it --rm resnet-torch-native
```

## Caveats

- The WebAssembly build is slow, possible improvements would be getting around the limitations of `-Wl,--whole-archive` for `libtorch.a` and improving caching in the CMake build.
- JIT optimizations like `torch.jit.freeze` decrease the model size and improve performance but seem to depend on MKL-DNN which is not being compiled by default. 
- Most of the included optimization libraries are disabled since they complicate the build process so don't expect native speeds.

# License
This project is licensed under the MIT License.
