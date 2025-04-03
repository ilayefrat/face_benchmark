import os
import multiprocessing
multiprocessing.set_start_method('fork')
from facesBenchmarkUtils import *
from models import *
from tasks import *
from datetime import datetime, date
from sklearn.metrics.pairwise import cosine_distances
import argparse

# TODO - separate the model_name to model architecture and model weights
# TODO - add the option to load the model from the model path

def run_single_model(model_name, export_path, layers_to_extract=None, model_path=None):
    #script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct relative paths for datasets
    #accuracy_pairs_path = os.path.join(script_dir, 'tests_datasets', 'inversion', 'lfw_test_pairs_300_inverted_same.csv')
    #accuracy_images_path = os.path.join(script_dir, 'tests_datasets', 'inversion', 'aligned_images')
    #correlation_pairs_path = os.path.join(script_dir, 'tests_datasets', 'similarity_perception', 'faces_visual_perception_similarity_behavioral_summary.csv')
    #correlation_images_path = os.path.join(script_dir, 'tests_datasets', 'similarity_perception', 'aligned_images')
    # Dictionary mapping model names to their constructors and default layers
    model_constructors = {
        'VGG16': Vgg16Model,
        'DINO': DinoModel,
        'CLIP': CLIPModel,
        'RESNET': ResNetModel
    }

    if model_name not in model_constructors:
        raise ValueError(f"Unsupported model: {model_name}")

    # Get the constructor and default layers for the model
    constructor = model_constructors[model_name]
    temp_model_instance = constructor(model_name=model_name)

    # Get available layers from the model
    available_layers = [name for name, _ in temp_model_instance.model.named_modules()]

    if layers_to_extract is None:
        # Use default layers if none are specified
        print(f"No layers specified. Using default layers: {available_layers[-2:]}")
        layers_to_extract = available_layers[-2:]  # Example: Use last two layers as default
    else:
        # Check if requested layers exist in the model
        for layer in layers_to_extract:
            if layer not in available_layers:
                raise ValueError(f"Layer '{layer}' is not supported by {model_name}. Available layers: {available_layers}")

    # Initialize the model
    if model_path is None:
        model_instance = constructor(model_name=model_name, layers_to_extract=layers_to_extract)
    else:
        model_instance = constructor(model_name=model_name, weights_file_path=model_path, layers_to_extract=layers_to_extract)

    #directories
    lfw_pairs = './tests_datasets/LFW/lfw_test_pairs_only_img_names.txt'
    lfw_images = './tests_datasets/LFW/lfw-align-128'
    upright_path = './tests_datasets/inversion/lfw_test_pairs_300_upright_same.csv'
    inverted_path = './tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv'
    inversion_images_folder_path = './tests_datasets/inversion/stimuliLabMtcnn/'
    sim_international_memory_pairs = './tests_datasets/similarity_perception_international_celebs/faces_memory_visual_similarity_behavioral_summary.csv'
    sim_international_pairs = './tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary.csv'
    sim_international_DP_pairs = './tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary_DP.csv'
    sim_international_SP_pairs = './tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary_SP.csv'
    sim_international_images_folder_path = './tests_datasets/similarity_perception_international_celebs/intFacesLabMtcnn'
    sim_il_familiar_pairs = './tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_familiar_distances.csv'
    sim_il_unfamiliar_pairs = './tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_unfamiliar_distances.csv'
    sim_il_images_folder_path = './tests_datasets/similarity_perception_israeli_celebs/newIsraeliFacesStimuliLabMtcnn'
    caucasian_pairs_path = './tests_datasets/other_race/vggface_other_race_same_caucasian.csv'
    asian_pairs_path = './tests_datasets/other_race/vggface_other_race_same_asian.csv'
    other_race_images_folder_path = './tests_datasets/other_race/other_raceLabMtcnn'
    thatcher_combined_pairs = './tests_datasets/thatcher/human_ratings_thatcher_combined.csv'
    thatcher_images_folder_path = './tests_datasets/thatcher/images_thatcher_mtcnn'
    conditioned_pairs = './tests_datasets/critical_features/critical_features_all_conditions.csv'
    critical_distances_pairs = './tests_datasets/critical_features/critical_features_critical_distances.csv'
    noncritical_distances_pairs = './tests_datasets/critical_features/critical_features_noncritical_distances.csv'
    conditioned_images_folder_path = './tests_datasets/critical_features/img_dataset/joined'
    # Define tasks to run
    tasks = [
        AccuracyTask(
            task_name='LFW',
            pairs_file_path=lfw_pairs,
            images_folder_path=lfw_images,
            distance_function=cosine_distances,
            true_label='same'
        ),
        AccuracyTask(
            pairs_file_path = upright_path,
            images_folder_path = inversion_images_folder_path,
            true_label = 'same',
            distance_function = cosine_distances,
            task_name = 'Inversion Effect - Upright'
            ),
        AccuracyTask(
            pairs_file_path = inverted_path,
            images_folder_path = inversion_images_folder_path,
            true_label = 'same',
            distance_function = cosine_distances,
            task_name = 'Inversion Effect - Inverted'
            ),
        # CorrelationTask(
        #     pairs_file_path = sim_international_pairs,
        #     images_folder_path = sim_international_images_folder_path,
        #     task_name = 'International Celebs - Visual Perception Similarity',
        #     distance_function = cosine_distances,
        #     correlation_function = np.corrcoef
        #     ),
        # CorrelationTask(
        #     pairs_file_path = sim_international_memory_pairs,
        #     images_folder_path = sim_international_images_folder_path,
        #     task_name = 'International Celebs - Memory Perception Similarity',
        #     distance_function = cosine_distances,
        #     correlation_function = np.corrcoef
        #     ),
        # CorrelationTask(
        #     pairs_file_path = sim_international_DP_pairs,
        #     images_folder_path = sim_international_images_folder_path,
        #     task_name = 'International Celebs - Visual Perception Similarity DP',
        #     distance_function = cosine_distances,
        #     correlation_function = np.corrcoef
        #     ),
        # CorrelationTask(
        #     pairs_file_path = sim_international_SP_pairs,
        #     images_folder_path = sim_international_images_folder_path,
        #     task_name = 'International Celebs - Visual Perception Similarity SP',
        #     distance_function = cosine_distances,
        #     correlation_function = np.corrcoef
        #     ),
        # CorrelationTask(
        #     pairs_file_path = sim_il_familiar_pairs,
        #     images_folder_path = sim_il_images_folder_path,
        #     task_name = 'IL Celebs - Familiar Performance',
        #     distance_function = pairwise.cosine_distances,
        #     correlation_function = np.corrcoef 
        # ),
        # CorrelationTask(
        #     pairs_file_path = sim_il_unfamiliar_pairs,
        #     images_folder_path = sim_il_images_folder_path,
        #     task_name = 'IL Celebs - Unfamiliar Performance',
        #     distance_function = pairwise.cosine_distances,
        #     correlation_function = np.corrcoef 
        # ),
        AccuracyTask(
            pairs_file_path = caucasian_pairs_path,
            images_folder_path = other_race_images_folder_path,
            true_label = 'same',
            distance_function = pairwise.cosine_distances,
            task_name = 'Other Race Effect - Caucasian'
        ),
        AccuracyTask(
            pairs_file_path = asian_pairs_path,
            images_folder_path = other_race_images_folder_path,
            true_label = 'same',
            distance_function = pairwise.cosine_distances,
            task_name = 'Other Race Effect - Asian'
        )
        # RelativeDifferenceTask(
        #     pairs_file_path = thatcher_combined_pairs,
        #     images_folder_path = thatcher_images_folder_path,
        #     group_column = 'cond',
        #     distance_function = torch.cdist,
        #     task_name = 'Thatcher Effect'
        # ),
        # ConditionedAverageDistances(
        #     pairs_file_path = conditioned_pairs,
        #     images_folder_path = conditioned_images_folder_path,
        #     condition_column = 'cond',
        #     distance_function = pairwise.cosine_distances,
        #     normalize = True,
        #     task_name = 'Critical Features'
        # ),
        # CorrelationTask(
        #     pairs_file_path = critical_distances_pairs,
        #     images_folder_path = conditioned_images_folder_path,
        #     task_name = 'Critical Features - Critical Distances',
        #     distance_function = pairwise.cosine_distances,
        #     correlation_function = np.corrcoef
        # ),
        # CorrelationTask(
        #     pairs_file_path = noncritical_distances_pairs,
        #     images_folder_path = conditioned_images_folder_path,
        #     task_name = 'Critical Features - Non-Critical Distances',
        #     distance_function = pairwise.cosine_distances,
        #     correlation_function = np.corrcoef
        # )
    ]

    # Create task manager
    manager = MultiModelTaskManager(
        models=[model_instance],
        tasks=tasks,
        batch_size=32
    )

    # Run tasks
    manager.run_all_tasks_all_models(export_path=export_path, print_log=True)

    # Export results
    manager.export_computed_metrics(export_path)
    manager.export_model_results_by_task(export_path)
    manager.export_unified_summary(export_path)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Run face recognition benchmark for a single model.')
    # parser.add_argument('--model', type=str, required=True, help='Model to use (e.g., VGG16, DINO, CLIP, RESNET)')
    # parser.add_argument('--export-path', type=str, required=True, help='Path to export the results')
    # parser.add_argument('--layers-to-extract', type=str, nargs='+', default=None, help='List of layers to extract (optional)')
    # parser.add_argument('--model-path', type=str, default=None, help='Path to model weights file (optional)')

    # args = parser.parse_args()

    # run_single_model(
    #     model_name=args.model,
    #     export_path=args.export_path,
    #     layers_to_extract=args.layers_to_extract,
    #     model_path=args.model_path
    # )

    exp_id_tradeoff_MR = True
    if exp_id_tradeoff_MR:
        from glob import glob
        all_ids = [1000, 500, 200, 100, 50, 10, 5, 2]
        all_img_per_id = [300] #, 200, 100, 50, 20, 10, 5, 1]

        for ids in all_ids:
            for img_per_id in all_img_per_id:
                model_paths = glob(
                    '/home/hdd_storage/MR/results/asians/%(ids)d_ids/small_ids_no_asians_%(ids)d_ids_%(ids)d_%(img_per_id)d_*/vgg16/models/119.pth' % {
                        "ids": ids, "img_per_id": img_per_id})
                for model_path_index in range(len(model_paths)):
                    prefix = model_paths[model_path_index].split('/')[-4][len("small_ids_no_asians_"):]
                    output_folder = '/home/new_storage/experiments/seminar_benchmark/benchmark/outputs_ng/%s/' % prefix
                    print(model_paths[model_path_index])
                    print("output_folder", output_folder)
                    
                    run_single_model(
                        model_name="VGG16",
                        export_path=output_folder,
                        layers_to_extract=['avgpool', 'classifier.5'],
                        model_path=model_paths[model_path_index]
                    )



#python3 run_benchmark_all_tasks.py --model RESNET --export-path output
#python3 run_benchmark_all_tasks.py --model VGG16 --export-path output --layers-to-extract avgpool classifier.3  example for specific layers

#python3 run_benchmark_all_tasks.py --model VGG16 --export-path output_vgg16 --model-path /home/new_storage/experiments/ng_typicality_project/ng_typicality_project/models/vgg16_only_caucasian_vggface2_ids_500_acc_81.pth
#python3 run_benchmark_all_tasks.py --model VGG16 --export-path output_vgg16_objects --model-path /home/new_storage/experiments/face_memory_task/models/face_trained_vgg16_119.pth