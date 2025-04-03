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

def plot_human_shading(
    ax,
    human_df,
    col_group_1,
    col_group_2,
    color_group_1,
    color_group_2,
    label_group_1="Group 1",
    label_group_2="Group 2",
    layer_low=20,
    layer_high=100
):
    if human_df is None:
        return

    try:
        group1_low = human_df[
            (human_df["Model Name"] == "Humans") &
            (human_df["Layer Name"] == layer_low)
        ][col_group_1].values[0]

        group1_high = human_df[
            (human_df["Model Name"] == "Humans") &
            (human_df["Layer Name"] == layer_high)
        ][col_group_1].values[0]

        group2_low = human_df[
            (human_df["Model Name"] == "Humans") &
            (human_df["Layer Name"] == layer_low)
        ][col_group_2].values[0]

        group2_high = human_df[
            (human_df["Model Name"] == "Humans") &
            (human_df["Layer Name"] == layer_high)
        ][col_group_2].values[0]

        ax.fill_between([-0.45, +0.45], group1_low, group1_high, color=color_group_1, alpha=0.2, label=label_group_1, zorder=100)
        ax.plot([-0.45, 0.45], [group1_low, group1_low], color="darkred", linewidth=1.5, zorder=6)
        ax.plot([-0.45, 0.45], [group1_high, group1_high], color="darkred", linewidth=1.5, zorder=6)
        ax.fill_between([0.55, 1.45], group2_low, group2_high, color=color_group_2, alpha=0.2, label=label_group_2, zorder=100)
        ax.plot([0.55, 1.45], [group2_low, group2_low], color="darkred", linewidth=1.5, zorder=6)
        ax.plot([0.55, 1.45], [group2_high, group2_high], color="darkred", linewidth=1.5, zorder=6)

    except IndexError:
        print(f"Warning: Could not find expected data for layers {layer_low} and {layer_high}.")


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
        same = get_score("Critical Features - same: Mean")
        noncrit = get_score("Critical Features - non_critical_changes: Mean")
        crit = get_score("Critical Features - critical_changes: Mean")
        diff = get_score("Critical Features - diff: Mean")
        view_same = get_score("View Invariant - Frontal - Frontal - Same: Mean")
        view_qleft = get_score("View Invariant - Frontal - Quarter Left: Mean")
        view_hleft = get_score("View Invariant - Frontal - Half Left: Mean")
        view_diff = get_score("View Invariant - Frontal - Frontal - Diff: Mean")


        fig = plt.figure(figsize=(25, 12))
        gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.8])  # 3 rows

        # Row 1: 4 subplots
        ax0 = fig.add_subplot(gs[0, 0])  
        ax1 = fig.add_subplot(gs[0, 1])  
        ax2 = fig.add_subplot(gs[0, 2])  
        ax3 = fig.add_subplot(gs[0, 3])  
        ax4 = fig.add_subplot(gs[0, 4])  

        # Row 2: 2 wider subplots
        ax5 = fig.add_subplot(gs[1, 0:2]) 
        ax6 = fig.add_subplot(gs[1, 3:5])  

        # Row 3: 1 centered subplot
        ax7 = fig.add_subplot(gs[2, 0:2])

        #Row 4: View Invariant subplot
        ax8 = fig.add_subplot(gs[2, 3:5])

        axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

        model_name = row['Model Name'].lower()
        layer_name = row['Layer Name'].replace('.', '')
        fig.suptitle(f"{model_name} {layer_name} summary plot".title(), fontsize=16)

        human_behavior_path = "/home/new_storage/experiments/seminar_benchmark/benchmark/human_behavior_filled_rounded.csv"

        if os.path.exists(human_behavior_path):
            human_df = pd.read_csv(human_behavior_path)
        else:
            print("[WARNING] Human data not found. Skipping overlay.")
            human_df = None

        axes[0].bar(["LFW"], [lfw_acc], color="red", zorder=2)
        axes[0].set_ylim(0, 1)
        axes[0].set_title("LFW Accuracy")
        axes[0].bar_label(axes[0].containers[0], fmt="%.2f")

        if human_df is None:
            return

        try:
            group1_low = human_df[
                (human_df["Model Name"] == "Humans") &
                (human_df["Layer Name"] == 20)
            ]["LFW: AUC"].values[0]

            group1_high = human_df[
                (human_df["Model Name"] == "Humans") &
                (human_df["Layer Name"] == 100)
            ]["LFW: AUC"].values[0]
            

            axes[0].fill_between([-0.45, 0.45], group1_low, group1_high, color="red", alpha=0.2, label="LFW Human")
            axes[0].plot([-0.45, 0.45], [group1_low, group1_low], color="darkred", linewidth=1.5, zorder=6)
            axes[0].plot([-0.45, 0.45], [group1_high, group1_high], color="darkred", linewidth=1.5, zorder=6)
            axes[0].set_xlim(-0.5, 0.5)
        except IndexError:
            print(f"Warning: Could not find expected data for layers {20} and {100}.")

    
        axes[1].bar(["Upright", "Inverted"], [inv_up, inv_inv], color=["red", "blue"], zorder=2)
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Inversion Effect Accuracy")
        axes[1].bar_label(axes[1].containers[0], fmt="%.2f")

        plot_human_shading(
        axes[1],
        human_df,
        "Inversion Effect - Inverted: Accuracy",
        "Inversion Effect - Upright: Accuracy",
        "red",
        "blue",
        label_group_1="Human",
        label_group_2="Human",
        layer_low=20,
        layer_high=100)


        axes[2].bar(["Upright - Inverted"], [inv_diff], color="green", zorder=2)
        axes[2].set_ylim(-0.5, 0.5)
        axes[2].set_title("Inversion Difference")
        axes[2].bar_label(axes[2].containers[0], fmt="%.2f")
        

        axes[3].bar(["Caucasian", "Asian"], [or_cauc, or_asian], color=["red", "purple"])
        axes[3].set_ylim(0, 1)
        axes[3].set_title("Other Race Effect Accuracy")
        axes[3].bar_label(axes[3].containers[0], fmt="%.2f")

        plot_human_shading(
        axes[3],
        human_df,
        "Other Race Effect - Caucasian: Accuracy",
        "Other Race Effect - Asian: Accuracy",
        "red",
        "purple",
        label_group_1="Human 20–100 Caucasian",
        label_group_2="Human 20–100 Asian",
        layer_low=20,
        layer_high=100)

        axes[4].bar(["Caucasian - Asian"], [or_diff], color="green", zorder=2)
        axes[4].set_ylim(-0.5, 0.5)
        axes[4].set_title("Other-Race Difference")
        axes[4].bar_label(axes[4].containers[0], fmt="%.2f")

        axes[5].bar(["Perception", "Memory"], [intl_vis, intl_mem], color=["blue", "red"], zorder=2)
        axes[5].set_ylim(-1, 1)
        axes[5].set_title("International Performance Correlation")
        axes[5].bar_label(axes[5].containers[0], fmt="%.2f")

        plot_human_shading(
        axes[5],
        human_df,
        "International Celebs - Visual Perception Similarity: Correlation Score",
        "International Celebs - Visual Perception Similarity DP: Correlation Score",
        "blue",
        "red",
        label_group_1="Human 20–100 perseption",
        label_group_2="Human 20–100 memory",
        layer_low=20,
        layer_high=100)

        axes[6].bar(["Familiar", "Unfamiliar"], [il_fam, il_unfam], color=["blue", "red"], zorder=2)
        axes[6].set_ylim(-1, 1)
        axes[6].set_title("IL Celebs Performance Correlation")
        axes[6].bar_label(axes[6].containers[0], fmt="%.2f")

        plot_human_shading(
        axes[6],
        human_df,
        "IL Celebs - Familiar Performance: Correlation Score",
        "IL Celebs - Unfamiliar Performance: Correlation Score",
        "blue",
        "red",
        label_group_1="Celebs 20–100 Familiar",
        label_group_2="Celebs 20–100 Unfamiliar",
        layer_low=20,
        layer_high=100)

        axes[7].bar(["Same", "NonCritical", "Critical", "Diff"], [same, noncrit, crit, diff], color=["blue", "gray", "green", "purple"], zorder=2)
        axes[7].set_ylim(0, 1)
        axes[7].set_title("Critical Features Means")
        axes[7].bar_label(axes[7].containers[0], fmt="%.2f")

        axes[8].bar(
            ["Same", "Quarter Left", "Half Left", "Different"],
            [view_same, view_qleft, view_hleft, view_diff],
            color=["blue", "orange", "green", "purple"], zorder=2
        )
        axes[8].set_ylim(0, 1)
        axes[8].set_title("View Invariant Means")
        axes[8].bar_label(axes[8].containers[0], fmt="%.2f")

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
    sim_il_images_folder_path = './tests_datasets/similarity_perception_israeli_celebs/newIsraeliFacesStimuliLabMtcnn'
    sim_il_familiar_pairs = './tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_familiar_distances.csv'
    sim_il_unfamiliar_pairs = './tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_unfamiliar_distances.csv'
    caucasian_pairs_path = './tests_datasets/other_race/vggface_other_race_same_caucasian.csv'
    asian_pairs_path = './tests_datasets/other_race/vggface_other_race_same_asian.csv'
    other_race_images_folder_path = './tests_datasets/other_race/other_raceLabMtcnn'
    thatcher_combined_pairs = './tests_datasets/thatcher/human_ratings_thatcher_combined.csv'
    thatcher_images_folder_path = './tests_datasets/thatcher/images_thatcher_mtcnn'
    conditioned_images_folder_path = './tests_datasets/critical_features/img_dataset'
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
        ),
        ConditionedAverageDistances(
            pairs_file_path = './tests_datasets/view_invariant/view_invariant_all_conditions.csv',
            images_folder_path = './tests_datasets/view_invariant/img_dataset',
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
python3 run_bm_all_tasks_ilay.py \
  --architecture VGG16

# 2. VGG16 | No weights | Specific layers (classifier.3, classifier.6)
python3 run_bm_all_tasks_ilay.py  \
  --architecture VGG16 \
  --layers-to-extract classifier.5 

# 3. VGG16 | With weights | Default layers
python3 run_bm_all_tasks_ilay.py \
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