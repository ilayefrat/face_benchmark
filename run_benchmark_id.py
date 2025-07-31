import os
import argparse
import multiprocessing
from datetime import datetime
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import cosine_distances
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from colors_config import colors, dark_variants
from visualization.visualizationAllTasks import generate_summary_plot

from facesBenchmarkUtils import *
from models import *
from tasks import *


# ------------ Architecture & Weight Management ------------

def get_model_constructor(architecture_name):
    model_constructors = {
        'VGG16': Vgg16Model,
        'DINO': DinoModel,
        'CLIP': CLIPModel,
        'RESNET50': ResNetModel,
        'IRESNET100': IResNetModel,
        'FACENET': FaceNetModel,
    }

    if architecture_name not in model_constructors:
        raise ValueError(f"Unsupported model: {architecture_name}")
    
    return model_constructors[architecture_name]


def inspect_available_layers(architecture_name):
    constructor = get_model_constructor(architecture_name)
    temp_model = constructor(model_name=architecture_name)
    return [name for name, _ in temp_model.model.named_modules()]


def build_model(architecture_name, layers_to_extract, model_name=None):
    constructor = get_model_constructor(architecture_name)
    return constructor(model_name=model_name or architecture_name, layers_to_extract=layers_to_extract)

def load_model_weights(model_instance, model_path):
    if model_path:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Remove 'module.' prefix if saved from DataParallel
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model_state_dict = model_instance.model.state_dict()
        compatible_state_dict = {}
        skipped_keys = []

        for k, v in state_dict.items():
            if k in model_state_dict:
                if model_state_dict[k].shape == v.shape:
                    compatible_state_dict[k] = v
                else:
                    skipped_keys.append((k, v.shape, model_state_dict[k].shape))
            else:
                skipped_keys.append((k, v.shape, None))

        if skipped_keys:
            print(f"[INFO] Skipped loading {len(skipped_keys)} layer(s) due to mismatch or absence:")
            for k, loaded_shape, model_shape in skipped_keys:
                print(f"   - {k}: checkpoint shape {loaded_shape}, model shape {model_shape}")

        model_instance.model.load_state_dict(compatible_state_dict, strict=False)

    return model_instance


# ------------ Benchmark Runner ------------

def run_single_model(architecture_name, model_path, export_path, layers_to_extract=None, model_name=None):
    # Get available layers
    available_layers = inspect_available_layers(architecture_name)

    if layers_to_extract is None:
        if args.architecture.upper() == 'FACENET':
            args.layers_to_extract = ['last_bn']
        else:
            layers_to_extract = available_layers[-2:]
            print(f"No layers specified. Using default layers: {layers_to_extract}")
    else:
        for layer in layers_to_extract:
            if layer not in available_layers:
                raise ValueError(f"Layer '{layer}' is not valid for {architecture_name}. Available layers: {available_layers}")

    # Build model and load weights
    model_instance = build_model(architecture_name, layers_to_extract, model_name)
    print(f"[INFO] Model loaded on device: {model_instance.device}")
    model_instance = load_model_weights(model_instance, model_path)

    # Paths
    lfw_pairs = './tests_datasets/LFW/lfw_test_pairs_only_img_names.txt'
    lfw_images = './tests_datasets/LFW/lfw-align-128'
    upright_path = './tests_datasets/inversion/lfw_test_pairs_300_upright_same.csv'
    inverted_path = './tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv'
    inversion_images_folder_path = './tests_datasets/inversion/stimuliLabMtcnn/'
    # sim_international_images_folder_path = './tests_datasets/similarity_perception_international_celebs/intFacesLabMtcnn' ## Old Images
    sim_international_images_folder_path = './tests_datasets/img_links/downloaded_images/International' ## New Images
    sim_international_pairs = './tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary.csv'
    sim_international_memory_pairs = './tests_datasets/similarity_perception_international_celebs/faces_memory_visual_similarity_behavioral_summary.csv'
    # sim_il_images_folder_path = './tests_datasets/similarity_perception_israeli_celebs/newIsraeliFacesStimuliLabMtcnn' ## Old Images
    sim_il_images_folder_path = './tests_datasets/img_links/downloaded_images/Israeli' ## New Images
    sim_il_familiar_pairs = './tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_familiar_distances.csv'
    sim_il_unfamiliar_pairs = './tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_unfamiliar_distances.csv'
    caucasian_pairs_path = './tests_datasets/other_race/vggface_other_race_same_caucasian.csv'
    asian_pairs_path = './tests_datasets/other_race/vggface_other_race_same_asian.csv'
    other_race_images_folder_path = './tests_datasets/other_race/other_raceLabMtcnn'
    conditioned_images_folder_path = './tests_datasets/critical_features/img_dataset'
    conditioned_pairs = './tests_datasets/critical_features/critical_features_all_conditions.csv'
    critical_distances_pairs = './tests_datasets/critical_features/critical_features_critical_distances.csv'
    noncritical_distances_pairs = './tests_datasets/critical_features/critical_features_noncritical_distances.csv'
    view_invariant_pairs = './tests_datasets/view_invariant/view_invariant_all_conditions.csv'
    view_invariant_path = './tests_datasets/view_invariant/img_dataset'

    # Tasks
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
        CorrelationTask(
            pairs_file_path = sim_international_pairs,
            images_folder_path = sim_international_images_folder_path,
            task_name = 'International Celebs - Visual Perception Similarity',
            distance_function = cosine_distances,
            correlation_function = np.corrcoef
            ),
        CorrelationTask(
            pairs_file_path = sim_international_memory_pairs,
            images_folder_path = sim_international_images_folder_path,
            task_name = 'International Celebs - Memory Perception Similarity',
            distance_function = cosine_distances,
            correlation_function = np.corrcoef
            ),
        CorrelationTask(
            pairs_file_path = sim_il_familiar_pairs,
            images_folder_path = sim_il_images_folder_path,
            task_name = 'IL Celebs - Familiar Performance',
            distance_function = pairwise.cosine_distances,
            correlation_function = np.corrcoef 
        ),
        CorrelationTask(
            pairs_file_path = sim_il_unfamiliar_pairs,
            images_folder_path = sim_il_images_folder_path,
            task_name = 'IL Celebs - Unfamiliar Performance',
            distance_function = pairwise.cosine_distances,
            correlation_function = np.corrcoef 
        ),
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
        ),
        ConditionedAverageDistances(
            pairs_file_path = conditioned_pairs,
            images_folder_path = conditioned_images_folder_path,
            condition_column = 'cond',
            distance_function = pairwise.cosine_distances,
            normalize = True,
            task_name = 'Critical Features'
        ),
        CorrelationTask(
            pairs_file_path = critical_distances_pairs,
            images_folder_path = conditioned_images_folder_path,
            task_name = 'Critical Features - Critical Distances',
            distance_function = pairwise.cosine_distances,
            correlation_function = np.corrcoef
        ),
        CorrelationTask(
            pairs_file_path = noncritical_distances_pairs,
            images_folder_path = conditioned_images_folder_path,
            task_name = 'Critical Features - Non-Critical Distances',
            distance_function = pairwise.cosine_distances,
            correlation_function = np.corrcoef
        ),
        ConditionedAverageDistances(
            pairs_file_path = view_invariant_pairs, 
            images_folder_path = view_invariant_path, 
            condition_column = 'cond',
            distance_function = pairwise.cosine_distances,
            normalize = True,
            task_name = 'View Invariant'
        )
    ]
    # Manager and execution
    manager = MultiModelTaskManager(
        models=[model_instance],
        tasks=tasks,
        batch_size=32
    )

    manager.run_all_tasks_all_models(export_path=export_path, print_log=True)
    manager.export_computed_metrics(export_path)
    manager.export_model_results_by_task(export_path)
    manager.export_unified_summary(export_path)

    generate_summary_plot(export_path)  # Auto-generate visualization

def generate_export_path(args):
    """Generate a meaningful export folder name inside outputs/all_tasks/"""
    parts = [args.architecture.lower()]

    if args.model_path:
        parts.append("weights")
    else:
        parts.append("noweights")

    if args.model_name and args.architecture.upper().startswith("IRESNET"):
        parts.append(args.model_name.lower())

    if args.layers_to_extract:
        cleaned_layers = [layer.replace(".", "") for layer in args.layers_to_extract]
        parts.append("__".join(cleaned_layers))
    else:
        parts.append("defaultlayers")

    parts.append("all_tasks")

    return os.path.join("AllTasks", "_".join(parts))



# ------------ Entry Point ------------

if __name__ == '__main__':
    multiprocessing.set_start_method('fork')

    parser = argparse.ArgumentParser(description='Run face recognition benchmark for a single model.')
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--export-path', type=str, default=None)
    parser.add_argument('--layers-to-extract', type=str, nargs='+', default=None)
    parser.add_argument('--model-name', type=str, default=None)

    args = parser.parse_args()

    if args.layers_to_extract is None:
        args.layers_to_extract = inspect_available_layers(args.architecture)[-2:]
        print(f"[INFO] No layers specified. Using default layers: {args.layers_to_extract}")

    for single_layer in args.layers_to_extract:
        print(f"\n[INFO] Running benchmark for layer: {single_layer}")
        layer_args = argparse.Namespace(
            architecture=args.architecture,
            model_path=args.model_path,
            export_path=None,
            layers_to_extract=[single_layer],
            model_name=args.model_name
        )
        layer_args.export_path = generate_export_path(layer_args)

        run_single_model(
            architecture_name=layer_args.architecture,
            model_path=layer_args.model_path,
            export_path=layer_args.export_path,
            layers_to_extract=layer_args.layers_to_extract,
            model_name=layer_args.model_name
        )


"""
# ==========================
# Example Commands to Run
# ==========================

# 1. VGG16 | No weights | Default layers
python3 run_benchmark_id.py \
  --architecture VGG16

# 2. VGG16 | No weights | Specific layers (classifier.3, classifier.6)
python3 run_benchmark_id.py  \
  --architecture VGG16 \
  --layers-to-extract classifier.5 

# 3. VGG16 | With weights | Default layers
python3 run_benchmark_id.py \
  --architecture VGG16 \
  --model-path /home/new_storage/experiments/face_memory_task/models/face_trained_vgg16_119.pth

# 4. VGG16 | With weights | Specific layers (classifier.3, classifier.6)
python3 run_benchmark_id.py \
  --architecture VGG16 \
  --model-path /home/new_storage/experiments/face_memory_task/models/face_trained_vgg16_119.pth \
  --layers-to-extract classifier.3 classifier.6

# 5. RESNET | With ArcFace weights | Default layers
python3 run_benchmark_id.py \
  --architecture RESNET \
  --model-path ./weights/resnet/arcface_resnet100.pth

# 6. RESNET | No weights | Default layers
python3 run_benchmark_id.py \
  --architecture RESNET

# 7. IRESNET | ArcFace weights | Default layer ("fc")
python3 run_benchmark_id.py \
  --architecture IRESNET100 \
  --model-name arcface \
  --model-path ./weights/arcface/ms1mv3_arcface_r100.pth \
  --layers-to-extract fc

# 8. IRESNET | CosFace weights | Default layer ("fc")
python3 run_benchmark_id.py \
  --architecture IRESNET100 \
  --model-name cosface \
  --model-path ./weights/cosface/glint360k_cosface_r100.pth \
  --layers-to-extract fc




# 11. FACENET | pretrained on VGGFace2 | Layer last_bn
python3 run_benchmark_id.py \
  --architecture FACENET \
  --model-name facenet \
  --layers-to-extract last_bn

# 12. FLIP | pretrained on FaceCaption-15M | Layer embedding
python3 run_benchmark_id.py \
  --architecture FLIP \
  --model-path weights/flip/FLIP-base.pth

"""