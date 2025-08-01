from facesBenchmarkUtils.baseTask import *
from typing import Any, Callable

class ConditionedAverageDistances(BaseTask):
    def __init__(
        self,
        task_name: str,
        pairs_file_path: str,
        images_folder_path: str,
        distance_function: Callable[[Any, Any], float],
        condition_column: str,
        normalize: bool = True
    ) -> None:
        """
        Initializes the ConditionedAverageDistances instance.

        Parameters
        ----------
        task_name : str
            The name of the task.
        pairs_file_path : str
            Path to the CSV file containing image pairs and condition labels.
        images_folder_path : str
            Path to the directory containing images.
        distance_function : callable
            Function to compute the distance between image embeddings.
        condition_column : str
            Column name in the pairs file distinguishing between the different conditions.
        normialize: bool
            Boolean parameter for normializing the computed distances, by deviding each distance with the max distance computed. Default is True.
        """
        super().__init__(
            task_name=task_name,
            pairs_file_path=pairs_file_path,
            images_folder_path=images_folder_path,
            distance_function=distance_function
        )
        self.normalize = normalize
        if condition_column not in self.pairs_df.columns:
            raise Exception(f'The pairs file must contain a "{condition_column}" column.')
        else:
            self.pairs_df.rename(columns = {condition_column:'condition'}, inplace=True)

    def compute_task_performance(self, pairs_distances_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the relative difference between two groups.

        Parameters
        ----------
        pairs_distances_df : pandas.DataFrame
            DataFrame containing the computed distances for image pairs.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the group means, relative difference, and distance metric name.
        """
        if self.normalize:
            max_distance = pairs_distances_df['model_computed_distance'].max()
            if max_distance != 0:
                pairs_distances_df['normalized_distance'] = pairs_distances_df['model_computed_distance'] / max_distance
        else:
            pairs_distances_df['normalized_distance'] = pairs_distances_df['model_computed_distance']

        avg_distances = pairs_distances_df.groupby(['condition'])['normalized_distance'].mean().reset_index()
        
        avg_distances.rename(columns={'normalized_distance': 'Mean', 'condition': 'Condition'}, inplace=True)
        
        avg_distances['Distance Metric'] = self.distance_function_name

        return avg_distances
    
    @staticmethod
    def plot(
        output_dir: str,
        performances: pd.DataFrame,
        *optional: Any
    ) -> None:
        """
        Generates and saves a bar plot for each condition.

        Parameters
        ----------
        output_dir : str
            Directory where the plot images will be saved.
        performances : pandas.DataFrame
            DataFrame containing performance metrics for each model and task.
        *optional : Any
            Additional optional arguments (not used).

        Returns
        -------
        None
        """
        performances['Model-Layer'] = performances.apply(create_model_layer_column, axis=1)
        for model_layer_name, model_layer_df in performances.groupby('Model-Layer'):
            PlotHelper.bar_plot(
                performances=model_layer_df,
                x='Condition',
                xlabel='Condition',
                y='Mean',
                ylabel='Average Distance',
                title_prefix=f'Average Distance Comparison Across Conditions - {model_layer_name}',
                output_dir=output_dir,
                file_name=f'average_distance_cond_comparison: {model_layer_name}'
            )