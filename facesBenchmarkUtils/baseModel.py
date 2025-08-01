from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image

class BaseModel(ABC):
    """
    An abstract base class representing a neural network model in the benchmarking framework.
    All specific model classes should inherit from this class.

    Attributes
    ----------
    name : str
        The name of the model.
    weights_file_path : str or None
        Path to the model's weights file. If None, default weights are used.
    layers_to_extract : list of str
        List of layer names from which to extract outputs.
    preprocess : callable
        Function to preprocess input images.
    hook_outputs : dict
        Dictionary to store outputs from hooked layers.
    model : torch.nn.Module
        The neural network model.
    device : torch.device
        The device (CPU or GPU) on which the model is placed.
    num_identities : int or None
        Number of identities (classes) in the model, set if weights are loaded.
    """

    def __init__(
        self,
        model_name: str,
        weights_file_path: Optional[str] = None,
        layers_to_extract: Optional[Union[str, List[str]]] = None,
        preprocess_function: Optional[Callable[[Any], Any]] = None
    ) -> None:
        """
        Initializes the BaseModel instance.

        Parameters
        ----------
        model_name : str
            The name of the model.
        weights_file_path : str, optional
            Path to the model's weights file. If None, default weights are used.
        layers_to_extract : str or list of str, optional
            Layer name(s) from which to extract outputs.
        preprocess_function : callable, optional
            Function to preprocess input images. If None, a default preprocessing is used.
        """
        self.set_preprocess_function(preprocess_function)
        self.hook_outputs = {}
        self.model_name = model_name
        if isinstance(layers_to_extract, list):
            self.layers_to_extract = layers_to_extract
        elif layers_to_extract:
            self.layers_to_extract = [layers_to_extract]
        else:
            self.layers_to_extract = []
        self.weights_file_path = weights_file_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_identities = self._set_num_identities() if self.weights_file_path else None
        self.model = None
        self._build_model()
        if weights_file_path:
            self._load_model()
        self.to()
        if self.model:
            self.model.eval()
        self._register_hooks()

    @abstractmethod
    def _build_model(self) -> None:
        """
        Abstract method to build the neural network model.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for the forward pass of the model.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to the model.

        Returns
        -------
        torch.Tensor
            The output tensor from the model.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclasses.
        """
        pass

    def _set_num_identities(self) -> int:
        """
        Determines the number of identities (classes) based on the loaded weights.

        Returns
        -------
        int
            Number of identities in the model.
        """
        checkpoint = torch.load(self.weights_file_path, map_location='cpu')
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        last_key = list(checkpoint.keys())[-1]
        shape = checkpoint[last_key].shape
        if len(shape) == 2:
            return shape[0]  # classifier weights [C, D]
        else:
            return None  # ArcFace-style embedding-only model

    def _load_model(self) -> None:
        """
        Loads the model weights from the specified path.
        """
        checkpoint = torch.load(self.weights_file_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        if self.model:
            self.model.load_state_dict(state_dict)

            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            self.to()
            self.model.eval()

    def get_layer_names(self, simplified: bool = False) -> None:
        """
        Prints the names of all layers in the model.

        Parameters
        ----------
        simplified : bool, optional
            If True, prints only the layer names. If False, also prints layer details.
        """
        if self.model:
            layers = dict(self.model.named_modules())
            for name, info in layers.items():
                print(f'{name}\n{info}\n' if not simplified else name)

    def _register_hooks(self) -> None:
        """
        Registers forward hooks on specified layers to capture their outputs.
        """
        if self.layers_to_extract:
            for layer_name in self.layers_to_extract:
                layer = self._get_layer(layer_name)
                if layer:
                    layer.register_forward_hook(self._get_hook_fn(layer_name))

    def _get_hook_fn(self, layer_name: str) -> Callable[[nn.Module, Any, Any], None]:
        """
        Creates a hook function to capture the output of a layer.

        Parameters
        ----------
        layer_name : str
            The name of the layer to hook.

        Returns
        -------
        callable
            The hook function.
        """
        def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
            self.hook_outputs[layer_name] = output
        return hook_fn

    def _get_layer(self, layer_name: str) -> Optional[nn.Module]:
        """
        Retrieves a layer from the model by its name.

        Parameters
        ----------
        layer_name : str
            The name of the layer.

        Returns
        -------
        torch.nn.Module or None
            The requested layer.

        Raises
        ------
        ValueError
            If the layer name is not found in the model.
        """
        if layer_name is None or self.model is None:
            return None
        modules = dict(self.model.named_modules())
        if layer_name in modules:
            return modules[layer_name]
        else:
            raise ValueError(f"Layer {layer_name} not found in the model.")

    def get_output(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Runs the model on the input tensor and retrieves outputs from specified layers.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The preprocessed input tensor.

        Returns
        -------
        dict
            A dictionary mapping layer names to their output tensors.
            If no hooks are registered, returns the default output.
        """
        self.hook_outputs = {}
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self._forward(input_tensor)
            if self.hook_outputs:
                outputs = {}
                for layer_name, out in self.hook_outputs.items():
                    if isinstance(out, tuple):
                        out = out[0]
                    outputs[layer_name] = out.detach().cpu().reshape(out.size(0), -1)
            else:
                if isinstance(output, dict):
                    outputs = {
                        k: v.detach().cpu().reshape(v.size(0), -1)
                        for k, v in output.items()
                    }
                else:
                    outputs = {'default': output.detach().cpu().reshape(output.size(0), -1)}
            return outputs

    def to(self) -> None:
        """
        Moves the model to the specified device (CPU or GPU).
        """
        if self.model:
            self.model.to(self.device)

    def set_preprocess_function(
        self, preprocess_function: Optional[Callable[[Any], Any]]
    ) -> None:
        """
        Sets the preprocessing function for input images.

        Parameters
        ----------
        preprocess_function : callable or None
            A function that preprocesses PIL images into tensors.
            If None, a default preprocessing function is used.
        """
        if preprocess_function is None:
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.preprocess = preprocess_function