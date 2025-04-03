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

from facesBenchmarkUtils import *
from models import *
from tasks import *


# ------------ Architecture & Weight Management ------------

def get_model_constructor(architecture_name):
    model_constructors = {
        'VGG16': Vgg16Model,
        'DINO': DinoModel,
        'CLIP': CLIPModel,
        'RESNET': ResNetModel
    }

    if architecture_name not in model_constructors:
        raise ValueError(f"Unsupported model: {architecture_name}")
    
    return model_constructors[architecture_name]


def inspect_available_layers(architecture_name):
    constructor = get_model_constructor(architecture_name)
    temp_model = constructor(model_name=architecture_name)
    return [name for name, _ in temp_model.model.named_modules()]


def build_model(architecture_name, layers_to_extract):
    constructor = get_model_constructor(architecture_name)
    return constructor(model_name=architecture_name, layers_to_extract=layers_to_extract)


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

# ------------ Benchmark Visualizer  ------------

def generate_summary_plot(export_path):
    summary_path = os.path.join(export_path, "models_unified_results.csv")
    if not os.path.exists(summary_path):
        print("[WARNING] No summary CSV found. Skipping visualization.")
        return
    summary = pd.read_csv(summary_path)
    print(f"[INFO] Generating {len(summary)} summary plot(s)...")
    for idx, row in summary.iterrows():
        get_score = lambda col: row[col]
        lfw_acc = get_score("LFW: Accuracy")
        inv_up = get_score("Inversion Effect - Upright: Accuracy")
        inv_inv = get_score("Inversion Effect - Inverted: Accuracy")
        inv_diff = inv_up - inv_inv
        or_cauc = get_score("Other Race Effect - Caucasian: Accuracy")
        or_asian = get_score("Other Race Effect - Asian: Accuracy")
        or_diff = or_cauc - or_asian
        intl_vis = get_score("International Celebs - Visual Perception Similarity: Correlation Score")
        intl_mem = get_score("International Celebs - Memory Perception Similarity: Correlation Score")
        il_fam = get_score("IL Celebs - Familiar Performance: Correlation Score")
        il_unfam = get_score("IL Celebs - Unfamiliar Performance: Correlation Score")
        dp = get_score("International Celebs - Visual Perception Similarity DP: Correlation Score")
        sp = get_score("International Celebs - Visual Perception Similarity SP: Correlation Score")
        same = get_score("Critical Features - same: Mean")
        noncrit = get_score("Critical Features - non_critical_changes: Mean")
        crit = get_score("Critical Features - critical_changes: Mean")
        diff = get_score("Critical Features - diff: Mean")

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        model_name = row['Model Name'].lower()
        layer_name = row['Layer Name'].replace('.', '')
        fig.suptitle(f"{model_name} {layer_name} summary plot".title(), fontsize=16)

        axes[0, 0].bar(["LFW"], [lfw_acc], color="red")
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_title("LFW Accuracy")
        axes[0, 0].bar_label(axes[0, 0].containers[0], fmt="%.2f")

        axes[0, 1].bar(["Upright", "Inverted"], [inv_up, inv_inv], color=["red", "blue"])
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title("Inversion Effect Accuracy")
        axes[0, 1].bar_label(axes[0, 1].containers[0], fmt="%.2f")

        axes[0, 2].bar(["Upright - Inverted"], [inv_diff], color="green")
        axes[0, 2].set_ylim(-0.5, 0.5)
        axes[0, 2].set_title("Inversion Difference")
        axes[0, 2].bar_label(axes[0, 2].containers[0], fmt="%.2f")

        axes[1, 0].bar(["Caucasian", "Asian"], [or_cauc, or_asian], color=["red", "blue"])
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_title("Other Race Effect Accuracy")
        axes[1, 0].bar_label(axes[1, 0].containers[0], fmt="%.2f")

        axes[1, 1].bar(["Caucasian - Asian"], [or_diff], color="green")
        axes[1, 1].set_ylim(-0.5, 0.5)
        axes[1, 1].set_title("Other-Race Difference")
        axes[1, 1].bar_label(axes[1, 1].containers[0], fmt="%.2f")

        axes[1, 2].bar(["Perception", "Memory"], [intl_vis, intl_mem], color=["blue", "red"])
        axes[1, 2].set_ylim(-1, 1)
        axes[1, 2].set_title("International Performance Correlation")
        axes[1, 2].bar_label(axes[1, 2].containers[0], fmt="%.2f")

        axes[2, 0].bar(["Familiar", "Unfamiliar"], [il_fam, il_unfam], color=["blue", "red"])
        axes[2, 0].set_ylim(-1, 1)
        axes[2, 0].set_title("IL Celebs Performance Correlation")
        axes[2, 0].bar_label(axes[2, 0].containers[0], fmt="%.2f")

        axes[2, 1].bar(["DP", "SP"], [dp, sp], color=["blue", "red"])
        axes[2, 1].set_ylim(-1, 1)
        axes[2, 1].set_title("DP-SP Performance Correlation")
        axes[2, 1].bar_label(axes[2, 1].containers[0], fmt="%.2f")

        axes[2, 2].bar(["Same", "NonCritical", "Critical", "Diff"], [same, noncrit, crit, diff], color=["blue", "gray", "green", "purple"])
        axes[2, 2].set_ylim(0, 1)
        axes[2, 2].set_title("Critical Features Means")
        axes[2, 2].bar_label(axes[2, 2].containers[0], fmt="%.2f")

        plt.subplots_adjust(top=0.9)
        filename = f"summary_plot_{model_name}_{layer_name}.png"
        output_path = os.path.join(export_path, filename)
        plt.savefig(output_path)
        plt.close(fig)
        print(f"[INFO] Saved summary visualization to {output_path}")


# ------------ Benchmark Runner ------------

def run_single_model(architecture_name, model_path, export_path, layers_to_extract=None):
    # Get available layers
    available_layers = inspect_available_layers(architecture_name)

    if layers_to_extract is None:
        layers_to_extract = available_layers[-2:]
        print(f"No layers specified. Using default layers: {layers_to_extract}")
    else:
        for layer in layers_to_extract:
            if layer not in available_layers:
                raise ValueError(f"Layer '{layer}' is not valid for {architecture_name}. Available layers: {available_layers}")

    # Build model and load weights
    model_instance = build_model(architecture_name, layers_to_extract)
    print(f"[INFO] Model loaded on device: {model_instance.device}")
    model_instance = load_model_weights(model_instance, model_path)

    # Paths
    lfw_pairs = './tests_datasets/LFW/lfw_test_pairs_only_img_names.txt'
    lfw_images = './tests_datasets/LFW/lfw-align-128'
    upright_path = './tests_datasets/inversion/lfw_test_pairs_300_upright_same.csv'
    inverted_path = './tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv'
    inversion_images_folder_path = './tests_datasets/inversion/stimuliLabMtcnn/'
    sim_international_images_folder_path = './tests_datasets/similarity_perception_international_celebs/intFacesLabMtcnn'
    sim_international_pairs = './tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary.csv'
    sim_international_memory_pairs = './tests_datasets/similarity_perception_international_celebs/faces_memory_visual_similarity_behavioral_summary.csv'
    sim_international_DP_pairs = './tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary_DP.csv'
    sim_international_SP_pairs = './tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary_SP.csv'
    sim_il_images_folder_path = './tests_datasets/similarity_perception_israeli_celebs/newIsraeliFacesStimuliLabMtcnn'
    sim_il_familiar_pairs = './tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_familiar_distances.csv'
    sim_il_unfamiliar_pairs = './tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_unfamiliar_distances.csv'
    caucasian_pairs_path = './tests_datasets/other_race/vggface_other_race_same_caucasian.csv'
    asian_pairs_path = './tests_datasets/other_race/vggface_other_race_same_asian.csv'
    other_race_images_folder_path = './tests_datasets/other_race/other_raceLabMtcnn'
    thatcher_combined_pairs = './tests_datasets/thatcher/human_ratings_thatcher_combined.csv'
    thatcher_images_folder_path = './tests_datasets/thatcher/images_thatcher_mtcnn'
    conditioned_images_folder_path = './tests_datasets/critical_features/img_dataset/joined'
    conditioned_pairs = './tests_datasets/critical_features/critical_features_all_conditions.csv'
    critical_distances_pairs = './tests_datasets/critical_features/critical_features_critical_distances.csv'
    noncritical_distances_pairs = './tests_datasets/critical_features/critical_features_noncritical_distances.csv'

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
            pairs_file_path = sim_international_DP_pairs,
            images_folder_path = sim_international_images_folder_path,
            task_name = 'International Celebs - Visual Perception Similarity DP',
            distance_function = cosine_distances,
            correlation_function = np.corrcoef
            ),
        CorrelationTask(
            pairs_file_path = sim_international_SP_pairs,
            images_folder_path = sim_international_images_folder_path,
            task_name = 'International Celebs - Visual Perception Similarity SP',
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
        RelativeDifferenceTask(
            pairs_file_path = thatcher_combined_pairs,
            images_folder_path = thatcher_images_folder_path,
            group_column = 'cond',
            distance_function = torch.cdist,
            task_name = 'Thatcher Effect'
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

    parts.append("weights" if args.model_path else "noweights")

    if args.layers_to_extract:
        cleaned_layers = [layer.replace(".", "") for layer in args.layers_to_extract]
        parts.append("__".join(cleaned_layers))
    else:
        parts.append("defaultlayers")

    parts.append("all_tasks")

    return os.path.join("outputs", "all_tasks", "_".join(parts))



# ------------ Entry Point ------------

if __name__ == '__main__':
    multiprocessing.set_start_method('fork')

    parser = argparse.ArgumentParser(description='Run face recognition benchmark for a single model.')
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--export-path', type=str, default=None)
    parser.add_argument('--layers-to-extract', type=str, nargs='+', default=None)

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
        )
        layer_args.export_path = generate_export_path(layer_args)

        run_single_model(
            architecture_name=layer_args.architecture,
            model_path=layer_args.model_path,
            export_path=layer_args.export_path,
            layers_to_extract=layer_args.layers_to_extract
        )


"""
# ==========================
# Example Commands to Run
# ==========================

# 1. VGG16 | No weights | Default layers
python3 run_benchmark_all_tasks_stav.py \
  --architecture VGG16

# 2. VGG16 | No weights | Specific layers (classifier.3, classifier.6)
python3 run_benchmark_all_tasks_stav.py \
  --architecture VGG16 \
  --layers-to-extract classifier.3 classifier.6

# 3. VGG16 | With weights | Default layers
python3 run_benchmark_all_tasks_stav.py \
  --architecture VGG16 \
  --model-path /home/new_storage/experiments/face_memory_task/models/face_trained_vgg16_119.pth

# 4. VGG16 | With weights | Specific layers (classifier.3, classifier.6)
python3 run_benchmark_all_tasks_stav.py \
  --architecture VGG16 \
  --model-path /home/new_storage/experiments/face_memory_task/models/face_trained_vgg16_119.pth \
  --layers-to-extract classifier.3 classifier.6

# Output folders will be auto-generated in:
# outputs/all_tasks/vgg16_<weights/noweights>_<layers>_all_tasks
"""