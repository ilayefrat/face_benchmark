import torch
import torch.nn as nn
from torchvision import models
from facesBenchmarkUtils.baseModel import *
from typing import Optional, List, Union, Callable

class SimCLRModel(BaseModel):
    def __init__(
        self, 
        model_name: str,
        weights_file_path: Optional[str] = None, 
        layers_to_extract: Optional[Union[str, List[str]]] = '0.avgpool', 
        preprocess_function: Optional[Callable[[any], any]] = None):
        
        super().__init__(
            model_name=model_name,
            weights_file_path=weights_file_path, 
            layers_to_extract=layers_to_extract, 
            preprocess_function=preprocess_function)


    def _build_model(self):
        # Load the ResNet50 backbone
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if self.weights_file_path is None else None)
        
        # Get the number of features from the original fully connected (fc) layer
        num_features = backbone.fc.in_features
        
        # Replace the classification head with Identity (no-op)
        backbone.fc = nn.Identity()
        
        # Add the projection head for SimCLR
        self.model = nn.Sequential(
            backbone,
            nn.Linear(num_features, 128),  # First projection head layer
            nn.ReLU(),
            nn.Linear(128, 128)  # Final projection head layer
        )


    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if necessary
        return self.model(input_tensor)

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.preprocess(image)
