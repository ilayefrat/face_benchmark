from facesBenchmarkUtils.baseModel import *
from torchvision import models
import torch.nn as nn

class ResNetModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        weights_file_path: Optional[str] = None,
        layers_to_extract: Optional[Union[str, List[str]]] = 'avgpool',
        preprocess_function: Optional[Callable[[Any], Any]] = None
    ):
        """
        Initialize the ResNet model.

        Parameters:
        - name (str): Model name.
        - weights_file_path (str, optional): Path to pre-trained weights.
        - extract_layers (str or list, optional): Layer(s) to extract outputs from.
        - preprocess_function (callable, optional): Custom preprocessing function.
        """
        super().__init__(
            model_name=model_name,
            weights_file_path=weights_file_path,
            layers_to_extract=layers_to_extract,
            preprocess_function=preprocess_function
        )

    def _build_model(self):
        # Load ResNet50 with pre-trained weights (from torchvision)
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if self.weights_file_path is None else None)
        if self.num_identities is not None:
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_identities)
        else:
            self.num_identities = model.fc.out_features
        
        self.model = model

    def _forward(self, input_tensor):
        # Forward pass through the ResNet model
        if input_tensor.ndim == 3:  # If input is a single image
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        return self.model(input_tensor)

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.preprocess(image)
