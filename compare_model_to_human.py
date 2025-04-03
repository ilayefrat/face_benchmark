import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def generate_graphs(human_csv_path, model_csv_path, output_dir="output"):
    # Load human and model data
    human_data = pd.read_csv(human_csv_path)
    model_data = pd.read_csv(model_csv_path)

    # Define tasks based on human data
    human_tasks = human_data['Task']
    task_columns = [col for col in model_data.columns if col in human_tasks.values]

    # Reshape model data for easier comparison
    model_data_long = model_data.melt(id_vars=["Model Name", "Layer Name"], 
                                      value_vars=task_columns, 
                                      var_name="Task", 
                                      value_name="Model Performance")
    
    # Merge human and model data on tasks
    comparison_df = pd.merge(human_data, model_data_long, on="Task")
    
    # Extract model name for folder naming
    model_name = model_data["Model Name"].iloc[0]

    # Create the output directory
    output_folder = os.path.join(output_dir, f"{model_name} compared to human")
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate task weights
    comparison_df["Task Weight"] = 1 / comparison_df["Human Performance Std"]
    comparison_df["Task Weight"] /= comparison_df["Task Weight"].sum()  # Normalize weights
    
    # Calculate human-likeness scores per layer
    human_likeness = comparison_df.groupby(["Model Name", "Layer Name"]).apply(
        lambda x: (x["Task Weight"] * abs(x["Model Performance"] - x["Human Performance Mean"])).sum()
    ).reset_index(name="Human-Likeness Score")
    human_likeness["Human-Likeness Score"] = 1 - human_likeness["Human-Likeness Score"]  # Higher is better
    
    # Plot 1: Comparison graph
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_df, x="Task", y="Human Performance Mean", color="skyblue", label="Human Performance")
    sns.barplot(data=comparison_df, x="Task", y="Model Performance", color="orange", alpha=0.7, label="Model Performance")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Performance")
    plt.title("Human vs. Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    comparison_path = os.path.join(output_folder, "comparison_graph.png")
    plt.savefig(comparison_path)
    plt.close()
    
    # Plot 2: Human-likeness graph
    plt.figure(figsize=(10, 6))
    sns.barplot(data=human_likeness, x="Layer Name", y="Human-Likeness Score", hue="Model Name", palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Human-Likeness Score")
    plt.title("Human-Likeness by Model Layer")
    plt.legend(title="Model Name")
    plt.tight_layout()
    human_likeness_path = os.path.join(output_folder, "human_likeness_graph.png")
    plt.savefig(human_likeness_path)
    plt.close()
    
    print(f"Graphs saved to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate performance comparison and human-likeness graphs.")
    parser.add_argument("human_csv", help="Path to the human performance CSV file.")
    parser.add_argument("model_csv", help="Path to the model performance CSV file.")
    parser.add_argument("--output_dir", default="output", help="Directory to save the generated graphs.")

    args = parser.parse_args()

    generate_graphs(args.human_csv, args.model_csv, args.output_dir)


# Command line usage example:
# python compare_model_to_human.py human_data.csv vgg16_data.csv --output_dir output