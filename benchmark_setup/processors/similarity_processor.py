import os
import pandas as pd
import requests
import cv2
from benchmark_setup.utils.crop_faces import process_image

CATEGORY_TO_DIR = {
    "International": os.path.join("tests_data", "similarity_perception", "International_mem", "images"),
    "Israeli": os.path.join("tests_data", "similarity_perception", "Israeli_fam", "images"),
}

def download_and_crop_images_from_excel(excel_file, output_dirs):
    print("\n[INFO] Processing Similarity Task images (Israeli and International)...")

    df = pd.read_excel(excel_file)
    required_cols = {"url", "name", "category"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"[ERROR] Excel file must contain the following columns: {required_cols}")

    seen = set()
    for _, row in df.iterrows():
        url = row["url"]
        name = row["name"]
        category = row["category"]

        if category not in output_dirs:
            print(f"[WARNING] Unknown category '{category}', skipping.")
            continue

        if not any(name.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
            name += ".jpg"

        if name in seen:
            print(f"[DEBUG] Skipping duplicate: {name}")
            continue
        seen.add(name)

        output_dir = output_dirs[category]
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, name)

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"[WARNING] Failed to download {url}: {e}")
            continue

        processed = process_image(filepath)
        if processed is None:
            print(f"[!] No face detected in: {name}")
            continue

        cv2.imwrite(filepath, processed)
        print(f"[✓] Saved: {filepath}")

    print("[✓] Done processing similarity images.")
