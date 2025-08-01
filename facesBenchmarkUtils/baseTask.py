import os
import torch
from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise
from .plotHelper import *

class BaseTask(ABC):
    """
    An abstract base class to represent a task in the benchmarking framework.
    All specific task classes should inherit from this class.

    Attributes
    ----------
    task_name : str
        The name of the task.
    images_folder_path : str
        The directory where the images are stored.
    pairs_file_path : str
        The path to the CSV file containing image pairs.
    pairs_df : pandas.DataFrame
        DataFrame containing pairs of images and related information.
    distance_function : callable
        The function used to compute the distance between image embeddings.
    """

    def __init__(
        self,
        task_name: str,
        pairs_file_path: str,
        images_folder_path: str,
        distance_function: Callable[[Any, Any], float] = pairwise.cosine_distances
    ) -> None:
        """
        Initializes the BaseTask instance.

        Parameters
        ----------
        task_name : str
            The name of the task.
        pairs_file_path : str
            Path to the CSV file containing image pairs.
        images_folder_path : str
            Path to the directory containing images.
        distance_function : callable, optional
            Function to compute the distance between image embeddings. Default is cosine distance.

        Raises
        ------
        FileNotFoundError
            If the provided image path or pairs file path does not exist.
        Exception
            If the pairs file does not contain required columns.
        """
        self.task_name = task_name
        self.pairs_file_path = pairs_file_path
        self.pairs_df = self.__load_file(pairs_file_path)
        self.images_folder_path = self.__validate_path(images_folder_path)
        self.distance_function, self.distance_function_name = self.__validate_and_set_distance_function(distance_function)

    @abstractmethod
    def compute_task_performance(self, pairs_distances_df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to compute the task performance metrics.

        Parameters
        ----------
        pairs_distances_df : pandas.DataFrame
            DataFrame containing the computed distances for image pairs.

        Returns
        -------
        pd.DataFrame
            The performance metrics DataFrame for the task.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def plot(self,
        output_dir: str,
        performances: pd.DataFrame,
        *optional: Any):
        """
        Abstract and static method to plot task's performance results.
        Can be implemented using PlotHelper class or using the different plotting packages.

        Parameters
        ----------
        output_dir : str
            Path to save the resulted plots.
        performances : pandas.DataFrame
            DataFrame containing the computed metrics for all the taks of the same type.
        *optional : Any
            Any other parameter you might need to create the plot.
        """
        pass

    def __to_float(self, x: Any) -> float:
        """
        Converts input to a float value. Supports torch.Tensor, numpy.ndarray, and scalars.

        Parameters
        ----------
        x : torch.Tensor or numpy.ndarray or scalar
            The input value to convert.

        Returns
        -------
        float
            The converted float value.
        """
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.item()
            else:
                return x.mean().item()
        elif isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.item())
            else:
                return float(x.mean())
        elif np.isscalar(x):
            return float(x)
        else:
            return float(x)

    def __validate_and_set_distance_function(
            self, 
            user_func: Callable[[Any, Any], Any]
            ) -> Tuple[Callable[[Any, Any], float], str]:
        """
        Validates the user-provided distance metric function.

        Parameters
        ----------
        user_func : callable
            The distance metric function to validate.

        Returns
        -------
        callable
            The validated distance metric function that returns a float.

        Raises
        ------
        Exception
            If the distance metric function is invalid.
        """
        try:
            rand_t1 = torch.rand((10, 2))
            rand_t2 = torch.rand((10, 2))
            result = user_func(rand_t1, rand_t2)
            self.__to_float(result)
        except Exception as e:
            raise Exception("Distance metric is not valid!") from e

        try:
            self.__to_float(user_func(rand_t1, rand_t2))
        except Exception:
            print(
                "WARNING! The distance function does not return a scalar or an array. "
                "This could potentially affect computing. Please consider changing the function."
            )
        return lambda x, y: self.__to_float(user_func(x, y)), user_func.__name__

    def __validate_path(self, path: str) -> str:
        """
        Validates that the provided path exists.

        Parameters
        ----------
        path : str
            The file or directory path to validate.

        Returns
        -------
        str
            The validated path.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File Not Found! Please provide the full path of the file.\nPath provided: {path}"
            )
        return path

    def __load_file(self, pairs_file_path: str) -> pd.DataFrame:
        """
        Loads the pairs CSV file into a DataFrame.

        Parameters
        ----------
        pairs_file_path : str
            The path to the pairs CSV file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the image pairs.

        Raises
        ------
        Exception
            If the required columns are not present in the CSV file.
        """
        self.__validate_path(pairs_file_path)
        try:
            pairs_pd = pd.read_csv(pairs_file_path)
            if not {'img1', 'img2'}.issubset(pairs_pd.columns):
                raise Exception("The CSV file must contain 'img1' and 'img2' columns.")
            return pairs_pd
        except Exception as e:
            raise e