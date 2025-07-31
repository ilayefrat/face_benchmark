import os
import shutil
import pandas as pd

# Paths
csv_path = "tests_datasets/critical_features/critical_features_all_conditions.csv"
img_folder = "tests_datasets/critical_features/img_dataset"
output_folder = "tests_data/critical_features/images"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Filter conditions
valid_conditions = {"critical_changes", "non_critical_changes"}
filtered = df[df["cond"].isin(valid_conditions)]

# Copy only images in img2
copied = 0
for fname in filtered["img2"].unique():
    src_path = os.path.join(img_folder, fname)
    dst_path = os.path.join(output_folder, fname)

    if not os.path.isfile(src_path):
        print(f"[WARNING] Missing: {src_path}")
        continue

    shutil.copy(src_path, dst_path)
    print(f"[✓] Copied: {fname}")
    copied += 1

print(f"[✓] Finished. Total images copied: {copied}")

# python benchmark_setup/utils/extract_critical_noncritical_images.py
