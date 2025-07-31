import multiprocessing
multiprocessing.set_start_method('fork')
from models import *

from facesBenchmarkUtils.multiModelTaskManager import MultiModelTaskManager
from tasks import *
from sklearn.metrics.pairwise import cosine_distances
from visualization.visualization import generate_summary_plot_multi_model
import argparse


def get_task_by_name(task_name):
    all_tasks = {
        'LFW': AccuracyTask(
            task_name='LFW',
            pairs_file_path='./tests_datasets/LFW/lfw_test_pairs_only_img_names.txt',
            images_folder_path='./tests_datasets/LFW/lfw-align-128',
            true_label='same',
            distance_function=cosine_distances
        ),
        'Inversion_Upright': AccuracyTask(
            task_name='Inversion Effect - Upright',
            pairs_file_path='./tests_datasets/inversion/lfw_test_pairs_300_upright_same.csv',
            images_folder_path='./tests_datasets/inversion/stimuliLabMtcnn/',
            true_label='same',
            distance_function=cosine_distances
        ),
        'Inversion_Inverted': AccuracyTask(
            task_name='Inversion Effect - Inverted',
            pairs_file_path='./tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv',
            images_folder_path='./tests_datasets/inversion/stimuliLabMtcnn/',
            true_label='same',
            distance_function=cosine_distances
        ),
        'International_Celebs_Visual_Perception_Similarity': CorrelationTask(
            task_name='International Celebs - Visual Perception Similarity',
            pairs_file_path='./tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary.csv',
            images_folder_path='./tests_datasets/img_links/downloaded_images/International',
            distance_function=cosine_distances,
            correlation_function=np.corrcoef
        ),
        'International_Celebs_Memory_Perception_Similarity': CorrelationTask(
            task_name='International Celebs - Memory Perception Similarity',
            pairs_file_path='./tests_datasets/similarity_perception_international_celebs/faces_memory_visual_similarity_behavioral_summary.csv',
            images_folder_path='./tests_datasets/img_links/downloaded_images/International',
            distance_function=cosine_distances,
            correlation_function=np.corrcoef
        ),
        'IL_Celebs_Familiar_Performance': CorrelationTask(
            task_name='IL Celebs - Familiar Performance',
            pairs_file_path='./tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_familiar_distances.csv',
            images_folder_path='./tests_datasets/img_links/downloaded_images/Israeli',
            distance_function=pairwise.cosine_distances,
            correlation_function=np.corrcoef
        ),
        'IL_Celebs_Unfamiliar_Performance': CorrelationTask(
            task_name='IL Celebs - Unfamiliar Performance',
            pairs_file_path='./tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_unfamiliar_distances.csv',
            images_folder_path='./tests_datasets/img_links/downloaded_images/Israeli',
            distance_function=pairwise.cosine_distances,
            correlation_function=np.corrcoef
        ),
        'Other_Race_Effect_Caucasian': AccuracyTask(
            task_name='Other Race Effect - Caucasian',
            pairs_file_path='./tests_datasets/other_race/vggface_other_race_same_caucasian.csv',
            images_folder_path='./tests_datasets/other_race/other_raceLabMtcnn',
            true_label='same',
            distance_function=pairwise.cosine_distances
        ),
        'Other_Race_Effect_Asian': AccuracyTask(
            task_name='Other Race Effect - Asian',
            pairs_file_path='./tests_datasets/other_race/vggface_other_race_same_asian.csv',
            images_folder_path='./tests_datasets/other_race/other_raceLabMtcnn',
            true_label='same',
            distance_function=pairwise.cosine_distances
        ),
        'Critical_Features': ConditionedAverageDistances(
            task_name='Critical Features',
            pairs_file_path='./tests_datasets/critical_features/critical_features_all_conditions.csv',
            images_folder_path='./tests_datasets/critical_features/img_dataset',
            condition_column='cond',
            distance_function=pairwise.cosine_distances,
            normalize=True
        ),
        'Critical_Features_Critical_Distances': CorrelationTask(
            task_name='Critical Features - Critical Distances',
            pairs_file_path='./tests_datasets/critical_features/critical_features_critical_distances.csv',
            images_folder_path='./tests_datasets/critical_features/img_dataset',
            distance_function=pairwise.cosine_distances,
            correlation_function=np.corrcoef
        ),
        'Critical_Features_Non_Critical_Distances': CorrelationTask(
            task_name='Critical Features - Non-Critical Distances',
            pairs_file_path='./tests_datasets/critical_features/critical_features_noncritical_distances.csv',
            images_folder_path='./tests_datasets/critical_features/img_dataset',
            distance_function=pairwise.cosine_distances,
            correlation_function=np.corrcoef
        ),
        'View_Invariant': ConditionedAverageDistances(
            task_name='View Invariant',
            pairs_file_path='./tests_datasets/view_invariant/view_invariant_all_conditions.csv',
            images_folder_path='./tests_datasets/view_invariant/img_dataset',
            condition_column='cond',
            distance_function=pairwise.cosine_distances,
            normalize=True
        ),
    }

    if task_name not in all_tasks:
        raise ValueError(f"Unknown task name: {task_name}. Available tasks: {list(all_tasks.keys())}")
    return all_tasks[task_name]
def run_benchmark(task_name, models, export_path):
    # Initialize models
    model_instances = []
    for model_name in models:
        if model_name == 'VGG16':
            model_instances.append(Vgg16Model(model_name='VGG16'))
        elif model_name == 'DINO':
            model_instances.append(DinoModel(model_name='DINO'))
        elif model_name == 'CLIP':
            model_instances.append(CLIPModel(model_name='CLIP'))
        elif model_name == 'RESNET':
            model_instances.append(ResNetModel(model_name='RESNET'))
        elif model_name == 'SIMCLR':  
            model_instances.append(SimCLRModel(model_name='SIMCLR'))
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    task_instance = get_task_by_name(task_name)

    # Create task manager
    manager = MultiModelTaskManager(
        models=model_instances,
        tasks=[task_instance],
        batch_size=32
    )

    # Run tasks
    manager.run_all_tasks_all_models(export_path=export_path, print_log=True)

    # Export results
    manager.export_computed_metrics(export_path)
    manager.export_model_results_by_task(export_path)
    manager.export_unified_summary(export_path)
    generate_summary_plot_multi_model(export_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run face recognition benchmark.')
    parser.add_argument('--task', type=str, required=True, help='Task to run (e.g., accuracy, correlation)')
    parser.add_argument('--models', type=str, nargs='+', required=True, help='List of models to use (e.g., VGG16, DINO, CLIP, SIMCLR, RESNET)')
    parser.add_argument('--export-path', type=str, required=True, help='Path to export the results')

    args = parser.parse_args()

    run_benchmark(
        task_name=args.task,
        models=args.models,
        export_path=args.export_path
    )

#command to run the benchmark - python3 run_benchmark.py --task View_Invariant --models VGG16 RESNET DINO CLIP --export-path benchmark_view_inv