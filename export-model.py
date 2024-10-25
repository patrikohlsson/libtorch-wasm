import torch
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(torch.nn.Module):
    """
    A wrapper class for the ResNet50 model that includes preprocessing
    and an example image for inference.
    """
    def __init__(self):
        super().__init__()
        # Load the default pretrained weights for ResNet50
        weights = ResNet50_Weights.DEFAULT
        self.resnet50 = resnet50(weights=weights)
        self.preprocess = weights.transforms()
        
        # Read and preprocess the example image
        img = read_image('example.png')
        img = img[:3]  # Remove alpha channel if present
        example = self.preprocess(img).unsqueeze(0)
        
        # Register the preprocessed image as a buffer
        self.register_buffer('example_image', example)
    
    def forward(self):
        # Run the model on the example image
        logits = self.resnet50(self.example_image)
        probs = logits[0].softmax(0)
        scores, class_ids = probs.topk(5, dim=-1)
        return class_ids, scores

# Instantiate the model
model = ResNet50()

# Calculate and print the number of trainable parameters
model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'ResNet50 parameter count: {model_params}')
print('Serializing model...')

# Convert the model to TorchScript
model.eval()
jit_model = torch.jit.script(model)

# Make sure it works
with torch.no_grad():
    class_ids, scores = jit_model()
    # Get the class names from the weights metadata
    categories = ResNet50_Weights.DEFAULT.meta['categories']
    class_names = [categories[class_id] for class_id in class_ids]
    print(f'Class Names: {class_names}')
    print(f'Scores: {scores}')

print('Saving model...')
jit_model.save('resnet50.pt')
