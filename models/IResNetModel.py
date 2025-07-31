from facesBenchmarkUtils.baseModel import *
from PIL import Image
from torchvision import transforms
import torch
import os

from models.iresnet_backbone import iresnet100  # shared backbone

class IResNetModel(BaseModel):
    """
    Unified model for ArcFace, CosFace, and similar models using iresnet100.
    The architecture is identical — only the weights differ.
    """
    def __init__(
        self,
        model_name: str,  # Can be 'arcface', 'cosface', etc. (for metadata only)
        weights_file_path: Optional[str] = None,
        layers_to_extract: Optional[Union[str, List[str]]] = 'fc'
    ):
        full_model_name = f"IResNet100 {model_name.capitalize()}"
        super().__init__(model_name=full_model_name,
                        weights_file_path=weights_file_path,
                        layers_to_extract=layers_to_extract)

    def _build_model(self):
        # Always construct the base architecture
        self.model = iresnet100()

        # During layer inspection, the model is constructed without weights
        if self.weights_file_path is None:
            return  # Skip weight loading; only inspecting layer names

        # For real runs, enforce valid path
        if not os.path.exists(self.weights_file_path):
            raise ValueError(f"Weight file not found: {self.weights_file_path}")

        # Load weights
        state_dict = torch.load(self.weights_file_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _forward(self, input_tensor):
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        return self.model(input_tensor)

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return transform(img)
