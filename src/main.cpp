#include <iostream>                 // For standard input/output streams
#include <iomanip>                  // For input/output manipulators like setw
#include <torch/script.h>           // For torch::jit::script::Module and torch::jit::load()
#include <torch/utils.h>            // For torch::NoGradGuard
#include "labels.h"                 // For imagenet_labels array

extern "C" {
    // Load the model from file
    int load_model();
    // Run the model and print predictions
    void run_model();
}

// Unique pointer to the loaded ResNet50 model
std::unique_ptr<torch::jit::script::Module> resnet50 = nullptr;

// Loads the ResNet50 model from the file "resnet50.pt"
int load_model() {
    try {
        // Deserialize the ScriptModule from a file
        resnet50 = std::make_unique<torch::jit::script::Module>(torch::jit::load("resnet50.pt"));
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

// Runs the loaded model and prints the predictions
void run_model() {
    torch::NoGradGuard no_grad;  // Disable gradient calculation

    // Perform a forward pass, example.png is stored inside the model.
    auto output = resnet50->forward({}).toTuple();

    // Extract class IDs and scores from the output tuple
    auto class_ids = output->elements()[0].toTensor();
    auto scores = output->elements()[1].toTensor();

    std::cout << "Predictions:" << std::endl;

    // Iterate over the predictions and print them
    for (int i = 0; i < class_ids.size(0); ++i) {
        int class_id = class_ids[i].item<int>();
        float score = scores[i].item<float>();
        std::cout << std::setw(20) << imagenet_labels[class_id] << ": " << std::right << score << std::endl;
    }
}

int main() {
    if (load_model() == 0) {
        run_model();
    }
    return 0;
}
