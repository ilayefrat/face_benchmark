import os
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from visualization import generate_summary_plot_multi_model
from colors_config import colors, dark_variants, col_to_name
from config import  task_groups

def strip_task_prefix(label):
    # Prefer splitting on colon (e.g., "LFW: Accuracy")
    if ":" in label:
        return label.split(":", 1)[1].strip()

    # Otherwise, if dash exists, try to remove a prefix up to first dash
    if "-" in label:
        parts = label.split("-", 1)
        if len(parts) == 2 and len(parts[0].strip().split()) <= 3:  # avoid breaking useful names
            return parts[1].strip()

    # Otherwise, return as-is
    return label.strip()

def create_task_specific_summary_csv(csv_paths, task_name):
    if task_name not in task_groups:
        raise ValueError(f"Task '{task_name}' not found in task_groups.")

    relevant_columns = task_groups[task_name]
    combined_rows = []

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"[WARNING] File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            model_row = {
                "Model Name": row["Model Name"],
                "Layer Name": row["Layer Name"]
            }

            for col in relevant_columns:
                model_row[col] = row.get(col, float('nan'))

            combined_rows.append(model_row)

    if not combined_rows:
        print(f"[WARNING] No data found for task '{task_name}'.")
        return pd.DataFrame()  # return empty DataFrame

    return pd.DataFrame(combined_rows)

def generate_multi_task_visualizations(csv_paths, task, export_path):
    os.makedirs(export_path, exist_ok=True)
    tmp = create_task_specific_summary_csv(csv_paths, task)
    temp_csv_path = os.path.join(export_path, "models_unified_results.csv")
    tmp.to_csv(temp_csv_path, index=False)
    generate_summary_plot_multi_model(export_path)

def generate_multi_task_visualizations_loop(csv_paths, task_list, export_base_path):
    for task in task_list:
        # Create a clean subdirectory name for each task
        safe_task_name = task.replace(" ", "_").replace("/", "_")
        task_export_path = os.path.join(export_base_path, safe_task_name)

        print(f"[INFO] Processing task: {task}")
        generate_multi_task_visualizations(csv_paths, task, task_export_path)
        



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-paths", nargs="+", required=True, help="Paths to models_unified_results.csv files")
    parser.add_argument("--tasks", nargs="+", required=True, help="List of task names to visualize")
    parser.add_argument("--export-path", type=str, required=True, help="Base export directory for saving plots")

    args = parser.parse_args()

    generate_multi_task_visualizations_loop(args.csv_paths, args.tasks, args.export_path)

    """
    python3 visualization_for_multi_comper.py \
        --csv-paths /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/resnet50_noweights_avgpool_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/vgg16_noweights_classifier5_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/vgg16_weights_classifier5_all_tasks/models_unified_results.csv  /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/iresnet100_weights_cosface_fc_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/iresnet100_weights_arcface_fc_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/facenet_noweights_last_bn_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/clip_noweights_ln_post_all_tasks/models_unified_results.csv\
        --tasks "IL Celebs" "International Celebs" "Critical Features" "View Invariant" "Other Race Effect" "Inversion Effect" "LFW"\
        --export-path ./visualizations_for_presentation
    
        this file, takes csv paths, tasks and export paths, and creats single task comperison between models - depends on existing results.

    """

    """
    python3 visualization_for_multi_comper.py \
        --csv-paths  /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/vgg16_weights_classifier5_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/facenet_noweights_last_bn_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/iresnet100_weights_cosface_fc_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/iresnet100_weights_arcface_fc_all_tasks/models_unified_results.csv  /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/clip_noweights_ln_post_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/dino_noweights_norm_all_tasks/models_unified_results.csv\
        --tasks "IL Celebs" "International Celebs" "Critical Features" "View Invariant" "Other Race Effect" "Inversion Effect" "LFW"\
        --export-path ./visualizations_for_presentation_dino_no_objects
    
        this file, takes csv paths, tasks and export paths, and creats single task comperison between models - depends on existing results.

    """

    """
    python3 visualization_for_multi_comper.py \
        --csv-paths  /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/resnet50_noweights_avgpool_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/vgg16_noweights_classifier5_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/dino_noweights_norm_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/clip_noweights_ln_post_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/vgg16_weights_classifier5_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/facenet_noweights_last_bn_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/iresnet100_weights_cosface_fc_all_tasks/models_unified_results.csv /home/new_storage/experiments/seminar_benchmark/benchmark/AllTasksToProject/iresnet100_weights_arcface_fc_all_tasks/models_unified_results.csv\
        --tasks "IL Celebs" "International Celebs" "Critical Features" "View Invariant" "Other Race Effect" "Inversion Effect" "LFW"\
        --export-path ./visualizations_for_presentation_dino_clip_objects
    
        this file, takes csv paths, tasks and export paths, and creats single task comperison between models - depends on existing results.

    """


