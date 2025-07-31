import timm
import torch
from PIL import Image
from facesBenchmarkUtils.baseModel import BaseModel

class DinoModel(BaseModel):
    def __init__(self, model_name='DINO', weights_file_path=None, layers_to_extract=None):
        super().__init__(
            model_name=model_name,
            weights_file_path=weights_file_path,
            layers_to_extract=layers_to_extract or [],  # We’re not using hooks, so pass an empty list
        )

    def _build_model(self):
        # Load pretrained DINO ViT-Small
        self.model = timm.create_model('vit_small_patch16_224_dino', pretrained=True)
        self.model.to(self.device)

    def _forward(self, input_tensor):
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            features = self.model(input_tensor.to(self.device))
            return {"default": features}  # So BaseModel.get_output can handle it


    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.preprocess(image)
