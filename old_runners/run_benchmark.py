import multiprocessing
multiprocessing.set_start_method('fork')
from models.vgg16Model import Vgg16Model #import all the models
from models.dinoModel import DinoModel
from models.clipModel import CLIPModel
from models.resNetModel import ResNetModel
from models.simCLRModel import SimCLRModel
from tasks.accuracyTask import AccuracyTask
from facesBenchmarkUtils.multiModelTaskManager import MultiModelTaskManager
from sklearn.metrics.pairwise import cosine_distances
import argparse



def run_benchmark(task_name, models, pairs_file_path, images_folder_path, export_path):
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

    # Initialize task
    if task_name == 'accuracy':
        task_instance = AccuracyTask(
            task_name='Accuracy Evaluation',
            pairs_file_path=pairs_file_path,
            images_folder_path=images_folder_path,
            distance_function=cosine_distances,
            true_label='same'
        )
    elif task_name == 'correlation':
        from tasks.correlationTask import CorrelationTask
        task_instance = CorrelationTask(
            task_name='Correlation Evaluation',
            pairs_file_path=pairs_file_path,
            images_folder_path=images_folder_path,
            distance_function=cosine_distances
        )
    else:
        raise ValueError(f"Unsupported task: {task_name}")

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run face recognition benchmark.')
    parser.add_argument('--task', type=str, required=True, help='Task to run (e.g., accuracy, correlation)')
    parser.add_argument('--models', type=str, nargs='+', required=True, help='List of models to use (e.g., VGG16, DINO, CLIP, SIMCLR, RESNET)')
    parser.add_argument('--pairs-file', type=str, required=True, help='Path to the pairs CSV file')
    parser.add_argument('--images-folder', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--export-path', type=str, required=True, help='Path to export the results')

    args = parser.parse_args()

    run_benchmark(
        task_name=args.task,
        models=args.models,
        pairs_file_path=args.pairs_file,
        images_folder_path=args.images_folder,
        export_path=args.export_path
    )

#command to run the benchmark - python3 run_benchmark.py --task accuracy --models VGG16 DINO --pairs-file tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv --images-folder tests_datasets/inversion/aligned_images --export-path output
# with layers to extract python3 run_benchmark.py --task accuracy --models VGG16 RESNET --pairs-file tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv --images-folder tests_datasets/inversion/aligned_images --export-path output --layers-to-extract avgpool classifier.3
#python3 run_benchmark.py --task accuracy --models SIMCLR --pairs-file tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv --images-folder tests_datasets/inversion/stimuliLabMtcnn --export-path simCLR_check
#python3 run_benchmark.py --task accuracy --models CLIP VGG16 DINO RESNET SIMCLR --pairs-file tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv --images-folder tests_datasets/inversion/stimuliLabMtcnn --export-path all_models