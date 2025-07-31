from facenet_pytorch import InceptionResnetV1
from facesBenchmarkUtils.baseModel import BaseModel
from PIL import Image
from torchvision import transforms
import torch
import os

class FaceNetModel(BaseModel):
    def __init__(self, model_name='facenet', weights_file_path=None, layers_to_extract='last_bn'):
        full_model_name = f"InceptionResnetV1 {model_name.capitalize()}"
        super().__init__(
            model_name=full_model_name,
            weights_file_path=weights_file_path,
            layers_to_extract=layers_to_extract
        )

    def _build_model(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def _forward(self, input_tensor):
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(input_tensor.to(self.device))
            return {
                "last_bn": embedding,
                "logits": embedding,
            }

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((160, 160)),  # FaceNet expects 160x160
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        return transform(img)
