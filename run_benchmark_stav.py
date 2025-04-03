import multiprocessing
multiprocessing.set_start_method('fork')

import argparse
import os
import torch
from sklearn.metrics.pairwise import cosine_distances

from models.vgg16Model import Vgg16Model
from models.dinoModel import DinoModel
from models.clipModel import CLIPModel
from models.resNetModel import ResNetModel
from models.simCLRModel import SimCLRModel

from tasks.accuracyTask import AccuracyTask
from tasks.correlationTask import CorrelationTask
from facesBenchmarkUtils.multiModelTaskManager import MultiModelTaskManager


# ------------------------ Model Utils ------------------------

def get_model_constructor(name):
    constructors = {
        'VGG16': Vgg16Model,
        'DINO': DinoModel,
        'CLIP': CLIPModel,
        'RESNET': ResNetModel,
        'SIMCLR': SimCLRModel
    }
    if name not in constructors:
        raise ValueError(f"Unsupported model: {name}")
    return constructors[name]


def build_model(name, layers_to_extract):
    constructor = get_model_constructor(name)
    return constructor(model_name=name, layers_to_extract=layers_to_extract)


def load_model_weights(model_instance, model_path):
    skipped_keys = []

    if model_path:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model_state_dict = model_instance.model.state_dict()
        compatible_state_dict = {
            k: v for k, v in state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }

        skipped_keys = [
            (k, v.shape, model_state_dict.get(k, None).shape if k in model_state_dict else None)
            for k, v in state_dict.items() if k not in compatible_state_dict
        ]

        if skipped_keys:
            print(f"[INFO] Skipped {len(skipped_keys)} incompatible layer(s):")
            for k, loaded_shape, model_shape in skipped_keys:
                print(f"   - {k}: checkpoint shape {loaded_shape}, model shape {model_shape}")

        model_instance.model.load_state_dict(compatible_state_dict, strict=False)

    return model_instance, skipped_keys


def generate_export_path(args):
    model_parts = []
    model_paths_dict = {
        kv.split('=')[0].strip(): kv.split('=')[1].strip()
        for kv in (args.model_paths or [])
        if '=' in kv
    }

    for model in sorted(args.models):
        weight_flag = "weights" if model in model_paths_dict else "noweights"
        model_parts.append(f"{model.lower()}[{weight_flag}]")

    layers_part = (
        "__".join([l.replace(".", "") for l in args.layers_to_extract])
        if args.layers_to_extract else "defaultlayers"
    )

    task = args.task.lower()
    folder_name = f"{'__'.join(model_parts)}_{layers_part}_{task}"
    return os.path.join("outputs", "models_per_task", folder_name)

# ------------------------ Benchmark Logic ------------------------

def get_task_instance(task_name, pairs_file_path, images_folder_path):
    if task_name == 'accuracy':
        return AccuracyTask(
            task_name='Accuracy Evaluation',
            pairs_file_path=pairs_file_path,
            images_folder_path=images_folder_path,
            distance_function=cosine_distances,
            true_label='same'
        )
    elif task_name == 'correlation':
        return CorrelationTask(
            task_name='Correlation Evaluation',
            pairs_file_path=pairs_file_path,
            images_folder_path=images_folder_path,
            distance_function=cosine_distances
        )
    else:
        raise ValueError(f"Unsupported task: {task_name}")


def run_benchmark(task_name, model_names, pairs_file_path, images_folder_path, export_path, model_paths=None, layers_to_extract=None):
    model_paths = model_paths or {}
    model_instances = []

    for model_name in model_names:
        model = build_model(model_name, layers_to_extract)
        model_path = model_paths.get(model_name)
        model, _ = load_model_weights(model, model_path)
        model_instances.append(model)

    task_instance = get_task_instance(task_name, pairs_file_path, images_folder_path)

    manager = MultiModelTaskManager(
        models=model_instances,
        tasks=[task_instance],
        batch_size=32
    )

    manager.run_all_tasks_all_models(export_path=export_path, print_log=True)
    manager.export_computed_metrics(export_path)
    manager.export_model_results_by_task(export_path)
    manager.export_unified_summary(export_path)


# ------------------------ Entry Point ------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run face recognition benchmark.')
    parser.add_argument('--task', type=str, required=True, help='Task to run (e.g., accuracy, correlation)')
    parser.add_argument('--models', type=str, nargs='+', required=True, help='List of models to use (e.g., VGG16, DINO, CLIP, SIMCLR, RESNET)')
    parser.add_argument('--pairs-file', type=str, required=True, help='Path to the pairs CSV file')
    parser.add_argument('--images-folder', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--export-path', type=str, default=None, help='Path to export the results (optional)')
    parser.add_argument('--model-paths', nargs='*', help='Optional list of model_name=model_path')
    parser.add_argument('--layers-to-extract', type=str, nargs='+', default=None, help='List of layers to extract (optional)')

    args = parser.parse_args()

    model_paths_dict = {}
    if args.model_paths:
        for item in args.model_paths:
            if '=' in item:
                name, path = item.split('=', 1)
                model_paths_dict[name.strip()] = path.strip()

    if args.export_path is None:
        args.export_path = generate_export_path(args)
        print(f"[INFO] Auto-generated export path: {args.export_path}")

    run_benchmark(
        task_name=args.task,
        model_names=args.models,
        pairs_file_path=args.pairs_file,
        images_folder_path=args.images_folder,
        export_path=args.export_path,
        model_paths=model_paths_dict,
        layers_to_extract=args.layers_to_extract
    )

"""
# ========================================================
# Example Commands for VGG16 + RESNET (Mixed Configs)
# ========================================================

# 1. VGG16 + RESNET | No weights | Default layers
python3 run_benchmark_stav.py \
  --task accuracy \
  --models VGG16 RESNET \
  --pairs-file tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv \
  --images-folder tests_datasets/inversion/stimuliLabMtcnn

# ➤ Output:
# outputs/models_per_task/resnet[noweights]__vgg16[noweights]_defaultlayers_accuracy


# 2. VGG16 + RESNET | No weights | Specific layers
python3 run_benchmark_stav.py \
  --task accuracy \
  --models VGG16 RESNET \
  --pairs-file tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv \
  --images-folder tests_datasets/inversion/stimuliLabMtcnn \
  --layers-to-extract avgpool classifier.3

# ➤ Output:
# outputs/models_per_task/resnet[noweights]__vgg16[noweights]_avgpool__classifier3_accuracy


# 3. VGG16 + RESNET | VGG16 with weights, RESNET without | Default layers
python3 run_benchmark_stav.py \
  --task accuracy \
  --models VGG16 RESNET \
  --model-paths VGG16=/home/new_storage/experiments/face_memory_task/models/face_trained_vgg16_119.pth \
  --pairs-file tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv \
  --images-folder tests_datasets/inversion/stimuliLabMtcnn

# ➤ Output:
# outputs/models_per_task/resnet[noweights]__vgg16[weights]_defaultlayers_accuracy


# 4. VGG16 + RESNET | Both with weights | Specific layers
python3 run_benchmark_stav.py \
  --task accuracy \
  --models VGG16 RESNET \
  --model-paths VGG16=/home/new_storage/experiments/face_memory_task/models/face_trained_vgg16_119.pth \
                 RESNET=/home/new_storage/experiments/face_memory_task/models/face_trained_resnet.pth \
  --pairs-file tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv \
  --images-folder tests_datasets/inversion/stimuliLabMtcnn \
  --layers-to-extract avgpool classifier.3

# ➤ Output:
# outputs/models_per_task/resnet[weights]__vgg16[weights]_avgpool__classifier3_accuracy
"""