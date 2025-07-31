import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from colors_config import colors, dark_variants, col_to_name


def get_dark_variant(color_name):
    dark_map = dark_variants
    return dark_map.get(color_name, color_name)

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

        ax.fill_between([-0.50, +0.50], group1_low, group1_high, color=color_group_1, alpha=0.2, label=label_group_1, zorder=100)
        ax.plot([-0.50, 0.50], [group1_low, group1_low], color=get_dark_variant(color_group_1), linewidth=1.5, zorder=6)
        ax.plot([-0.50, 0.50], [group1_high, group1_high], color=get_dark_variant(color_group_1), linewidth=1.5, zorder=6)
        ax.fill_between([0.50, 1.50], group2_low, group2_high, color=color_group_2, alpha=0.2, label=label_group_2, zorder=100)
        ax.plot([0.50, 1.50], [group2_low, group2_low], color=get_dark_variant(color_group_2), linewidth=1.5, zorder=6)
        ax.plot([0.50, 1.50], [group2_high, group2_high], color=get_dark_variant(color_group_2), linewidth=1.5, zorder=6)

    except IndexError:
        print(f"Warning: Could not find expected data for layers {layer_low} and {layer_high}.")

def plot_human_shading_single_group(
    ax,
    human_df,
    column_name,
    color,
    label="Human",
    layer_low=20,
    layer_high=100,
    x_start=-0.50,
    x_end=0.50
):
    if human_df is None:
        return

    try:
        y_low = human_df[
            (human_df["Model Name"] == "Humans") &
            (human_df["Layer Name"] == layer_low)
        ][column_name].values[0]

        y_high = human_df[
            (human_df["Model Name"] == "Humans") &
            (human_df["Layer Name"] == layer_high)
        ][column_name].values[0]

        ax.fill_between([x_start, x_end], y_low, y_high, color=color, alpha=0.2, label=label, zorder=100)
        ax.plot([x_start, x_end], [y_low, y_low], color=get_dark_variant(color), linewidth=1.5, zorder=6)
        ax.plot([x_start, x_end], [y_high, y_high], color=get_dark_variant(color), linewidth=1.5, zorder=6)
        ax.set_xlim(-0.5, 0.5)

    except IndexError:
        print(f"Warning: Could not find expected data for layers {layer_low} and {layer_high} in column {column_name}.")

def plot_human_shading_four_groups(
    ax,
    human_df,
    column_names,
    colors,
    labels=None,
    x_positions=None,
    layer_low=20,
    layer_high=100
):
    if human_df is None or len(column_names) != 4 or len(colors) != 4:
        return

    if labels is None:
        labels = ["Human"] * 4
    if x_positions is None:
        x_positions = [0, 1, 2, 3]

    for i in range(4):
        try:
            y_low = human_df[
                (human_df["Model Name"] == "Humans") &
                (human_df["Layer Name"] == layer_low)
            ][column_names[i]].values[0]

            y_high = human_df[
                (human_df["Model Name"] == "Humans") &
                (human_df["Layer Name"] == layer_high)
            ][column_names[i]].values[0]

            ax.fill_between([x_positions[i]-0.50, x_positions[i]+0.50], y_low, y_high, color=colors[i], alpha=0.2, label=labels[i], zorder=100)
            ax.plot([x_positions[i]-0.50, x_positions[i]+0.50], [y_low, y_low], color=colors[i], linewidth=1.5, zorder=6)
            ax.plot([x_positions[i]-0.50, x_positions[i]+0.50], [y_high, y_high], color=colors[i], linewidth=1.5, zorder=6)

        except IndexError:
            print(f"Warning: Could not find data for {column_names[i]} at layers {layer_low} and {layer_high}.")


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


        #fig = plt.figure(figsize=(25, 12))
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(18, 10))

        # Row 1 (5 graphs evenly spaced)
        # ax0 = plt.axes([0.20, 0.65, 0.06, 0.25])
        # ax1 = plt.axes([0.29, 0.65, 0.14, 0.25])
        # ax2 = plt.axes([0.47, 0.65, 0.06, 0.25])
        # ax3 = plt.axes([0.57, 0.65, 0.14, 0.25])
        # ax4 = plt.axes([0.75, 0.65, 0.06, 0.25])
        ax0 = plt.axes([0.30, 0.65, 0.06, 0.25])
        ax1 = plt.axes([0.40, 0.65, 0.14, 0.25])
        ax3 = plt.axes([0.57, 0.65, 0.14, 0.25])

        # Row 2 (centered)
        ax5 = plt.axes([0.27, 0.36, 0.22, 0.23])
        ax6 = plt.axes([0.52, 0.36, 0.22, 0.23])

        # Row 3 (centered)
        ax7 = plt.axes([0.33, 0.05, 0.14, 0.23])
        ax8 = plt.axes([0.52, 0.05, 0.14, 0.23])

        axes = [ax0, ax1, ax3, ax5, ax6, ax7, ax8]

        model_full_name = row['Model Name']
        layer_name = row['Layer Name'].replace('.', '')

        architecture = model_full_name.split()[0].upper()
        model_name_part = model_full_name.split()[1] if " " in model_full_name else None

        if model_name_part:
            title_str = f"{architecture} ({model_name_part}) {layer_name} Summary Plot"
        else:
            title_str = f"{architecture} {layer_name} Summary Plot"

        fig.suptitle(title_str, fontsize=16, fontweight='bold')
        human_behavior_path = "/home/new_storage/experiments/seminar_benchmark/benchmark/human_behavior_filled_rounded.csv"

        if os.path.exists(human_behavior_path):
            human_df = pd.read_csv(human_behavior_path)
        else:
            print("[WARNING] Human data not found. Skipping overlay.")
            human_df = None

        axes[0].bar(["LFW"], [lfw_acc], color=colors["lfw"], zorder=2, width=0.9)
        axes[0].set_ylim(0, 1.1)
        axes[0].set_title("LFW Accuracy",fontsize=13 ,fontweight='bold')
        axes[0].bar_label(axes[0].containers[0], fmt="%.2f")
        plot_human_shading_single_group(axes[0], human_df, column_name="Inversion Effect - Upright: Accuracy", color=colors["lfw"], label="Human", layer_low=20, layer_high=100)
  
        axes[1].bar(["Upright", "Inverted"], [inv_up, inv_inv], color=[colors["inversion_upright"], colors["inversion_inverted"]],edgecolor='black', zorder=2, width=0.9)
        axes[1].set_ylim(0, 1.1)
        axes[1].set_title("Face Inversion",fontsize=13 ,fontweight='bold')
        axes[1].bar_label(axes[1].containers[0], fmt="%.2f")

        plot_human_shading(
        axes[1],
        human_df,
        "Inversion Effect - Upright: Accuracy",
        "Inversion Effect - Inverted: Accuracy",
        colors["inversion_upright"], colors["inversion_inverted"],
        label_group_1="Human",
        label_group_2="Human",
        layer_low=20,
        layer_high=100)


        # axes[2].bar(["Upright - Inverted"], [inv_diff], color=colors["inversion_diff"], zorder=2, width=0.9)
        # axes[2].set_ylim(-0.5, 0.8)
        # axes[2].set_title("Difference",fontsize=13 ,fontweight='bold')
        # axes[2].bar_label(axes[2].containers[0], fmt="%.2f")
        # plot_human_shading_single_group(axes[2], human_df, column_name="Inversion Effect - Inverted: Accuracy", color=colors["inversion_diff"], label="Human", layer_low=20, layer_high=100)
        

        axes[2].bar(["Caucasian", "Asian"], [or_cauc, or_asian], color=[colors["other_race_caucasian"], colors["other_race_asian"]], width=0.9)
        axes[2].set_ylim(0, 1.1)
        axes[2].set_title("Other Race - Accuracy",fontsize=13 ,fontweight='bold')
        axes[2].bar_label(axes[2].containers[0], fmt="%.2f")

        plot_human_shading(
        axes[2],
        human_df,
        "Other Race Effect - Caucasian: Accuracy",
        "Other Race Effect - Asian: Accuracy",
        colors["other_race_caucasian"], colors["other_race_asian"],
        label_group_1="Human 20–100 Caucasian",
        label_group_2="Human 20–100 Asian",
        layer_low=20,
        layer_high=100)

        # axes[4].bar(["Caucasian - Asian"], [or_diff], color=colors["other_race_diff"], zorder=2, width=0.9)
        # axes[4].set_ylim(-0.5, 0.5)
        # axes[4].set_title("Other-Race Difference",fontsize=13 ,fontweight='bold')
        # axes[4].bar_label(axes[4].containers[0], fmt="%.2f")
        # plot_human_shading_single_group(axes[4], human_df, column_name="Thatcher Effect: Relative Difference", color=colors["other_race_diff"], label="Human", layer_low=20, layer_high=100)

        axes[5].bar(["Perception", "Memory"], [intl_vis, intl_mem], color=[colors["intl_perception"], colors["intl_memory"]], zorder=2, width=0.9)
        axes[5].set_ylim(-1, 1.1)
        axes[5].set_title("Representation Similarity\nPerseption vs. Memory",fontsize=13 ,fontweight='bold')
        axes[5].bar_label(axes[5].containers[0], fmt="%.2f")

        plot_human_shading(
        axes[5],
        human_df,
        "International Celebs - Visual Perception Similarity: Correlation Score",
        "International Celebs - Memory Perception Similarity: Correlation Score",
        colors["intl_perception"], colors["intl_memory"],
        label_group_1="Human 20–100 perseption",
        label_group_2="Human 20–100 memory",
        layer_low=20,
        layer_high=100)

        axes[6].bar(["Familiar", "Unfamiliar"], [il_fam, il_unfam], color=[colors["il_familiar"], colors["il_unfamiliar"]], zorder=2, width=0.9)
        axes[6].set_ylim(-1, 1.1)
        axes[6].set_title("Representation Similarity\nFamiliar vs. Unfamiliar",fontsize=13 ,fontweight='bold')
        axes[6].bar_label(axes[6].containers[0], fmt="%.2f")

        plot_human_shading(
        axes[6],
        human_df,
        "IL Celebs - Familiar Performance: Correlation Score",
        "IL Celebs - Unfamiliar Performance: Correlation Score",
        colors["il_familiar"], colors["il_unfamiliar"],
        label_group_1="Celebs 20–100 Familiar",
        label_group_2="Celebs 20–100 Unfamiliar",
        layer_low=20,
        layer_high=100)

        axes[3].bar(["Same", "NonCritical", "Critical", "Diff"], [same, noncrit, crit, diff], color=[colors["critical_same"], colors["critical_noncritical"],colors["critical_critical"], colors["critical_diff"]], zorder=2, width=0.9)
        axes[3].set_ylim(0, 1.1)
        axes[3].set_title("Critical Features Means",fontsize=13 ,fontweight='bold')
        axes[3].bar_label(axes[3].containers[0], fmt="%.2f")
        plot_human_shading_four_groups(axes[3], human_df, column_names=["Critical Features - same: Mean", "Critical Features - non_critical_changes: Mean", "Critical Features - critical_changes: Mean",
        "Critical Features - diff: Mean"], colors=[colors["critical_same"], colors["critical_noncritical"],colors["critical_critical"], colors["critical_diff"]], labels=["Same", "NonCritical", "Critical", "Diff"])

        axes[4].bar(
            ["Same", "Quarter Left", "Half Left", "Different"],
            [view_same, view_qleft, view_hleft, view_diff],
            color=[colors["view_same"], colors["view_qleft"],colors["view_hleft"], colors["view_diff"]], zorder=2, width=0.9
        )
        axes[4].set_ylim(0, 1.1)
        axes[4].set_title("View Invariant Means",fontsize=13 ,fontweight='bold')
        axes[4].bar_label(axes[4].containers[0], fmt="%.2f")
        plot_human_shading_four_groups(axes[4],human_df,column_names=["View Invariant - Frontal - Frontal - Same : Mean", "View Invariant - Frontal - Quarter Left : Mean", "View Invariant - Frontal - Half Left: Mean",
        "View Invariant - Frontal - Frontal - Diff: Mean"], colors=[colors["view_same"], colors["view_qleft"],colors["view_hleft"], colors["view_diff"]], labels=["Same", "Quarter Left", "Half Left", "Different"])

        for i in range(9-2):
            for label in axes[i].get_xticklabels():
                label.set_fontweight('bold')
        plt.subplots_adjust(top=0.9)
        filename = f"summary_plot_{model_full_name.replace(' ', '_')}_{layer_name}.png"
        output_path = os.path.join(export_path, filename)
        plt.savefig(output_path)
        plt.close(fig)
        print(f"[INFO] Saved summary visualization to {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, required=True, help='Function to run')
    parser.add_argument('--export-path', type=str, required=True, help='Path to CSV folder')

    args = parser.parse_args()

    if args.func == 'generate_summary_plot':
        generate_summary_plot(args.export_path)

    else:
        raise ValueError(f"Unknown function: {args.func}")

# python3 visualization.py --func generate_summary_plot --export-path /home/new_storage/experiments/seminar_benchmark/benchmark/outputs/all_tasks/iresnet100_weights_cosface_fc_all_tasks