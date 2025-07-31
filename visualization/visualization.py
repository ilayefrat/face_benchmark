import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from colors_config import colors, dark_variants, col_to_name
from config import pretty_task_names, task_groups


def get_dark_variant(color_name):
    dark_map = dark_variants
    return dark_map.get(color_name, color_name)



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



def find_task_group(col):
    for group, cols in task_groups.items():
        if col in cols:
            return group
    return None

def generate_summary_plot_multi_model(export_path):
    summary_path = os.path.join(export_path, "models_unified_results.csv")
    if not os.path.exists(summary_path):
        print("[WARNING] No summary CSV found. Skipping visualization.")
        return

    summary = pd.read_csv(summary_path)
    if summary.empty:
        print("[WARNING] Summary CSV is empty.")
        return

    print(f"[INFO] Generating combined single-plot summary for {len(summary)} model(s)...")

    # Get metric columns
    metric_columns = [
        col for col in summary.columns
        if col not in ["Model Name", "Layer Name"] and pd.api.types.is_numeric_dtype(summary[col])
    ]
    task_name = find_task_group(metric_columns[0])
    if not task_name:
        print(f"[ERROR] Could not find task group for column: {metric_columns[0]}")
        return

    models = summary["Model Name"].tolist()
    num_models = len(models)
    num_tasks = len(metric_columns)

    # Setup bar positions
    x = np.arange(num_models)  # model positions on x-axis
    total_width = 0.8
    bar_width = total_width / num_tasks

    # Assign one color per task


    task_to_title = task_name
    fig, ax = plt.subplots(figsize=(max(8, num_models * 1.2), 6))
    i=0
    for col in task_groups.get(task_name, []):

        y = summary[col].tolist()
        task_name = pretty_task_names.get(col)
        offset = x - total_width/2 + i * bar_width + bar_width/2
        ax.bar(offset, y, width=bar_width, color=colors[col_to_name[col]], label=task_name, edgecolor='black')
        i+=1

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    title_str = f"{task_to_title} - Model Benchmark Comparison"
    ax.set_title(title_str, fontsize=16, fontweight='bold')
    ax.legend(title="Task", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    output_path = os.path.join(export_path, "summary_all_tasks_single_plot.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"[INFO] Saved single-plot summary to {output_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, required=True, help='Function to run')
    parser.add_argument('--export-path', type=str, required=True, help='Path to CSV folder')

    args = parser.parse_args()

    if args.func == 'generate_summary_plot':
        #generate_summary_plot(args.export_path)
        generate_summary_plot_multi_model(args.export_path)
    else:
        raise ValueError(f"Unknown function: {args.func}")

# python3 visualization.py --func generate_summary_plot --export-path /home/new_storage/experiments/seminar_benchmark/benchmark/outputilayNEW2