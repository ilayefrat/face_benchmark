import os
import cv2
import pandas as pd
from benchmark_setup.utils.crop_faces import process_image

def process_lfw_accuracy(lfw_root, txt_path):
    print("\n[INFO] Processing LFW dataset for Accuracy Task...")
    output_root = "./tests_data/lfw/images/"

    df = pd.read_csv(txt_path)
    image_paths = set(df['img1']).union(set(df['img2']))
    print(f"[INFO] Found {len(image_paths)} unique images to process.")

    skipped = 0
    saved = 0

    for rel_path in sorted(image_paths):
        src_path = os.path.join(lfw_root, rel_path)
        person_folder, image_name = os.path.split(rel_path)
        dst_folder = os.path.join(output_root, person_folder)
        dst_path = os.path.join(dst_folder, image_name)

        if not os.path.isfile(src_path):
            print(f"[WARNING] Missing file: {src_path}")
            continue

        os.makedirs(dst_folder, exist_ok=True)
        processed = process_image(src_path)
        if processed is None:
            print(f"[!] No face detected: {rel_path}")
            skipped += 1
            continue

        cv2.imwrite(dst_path, processed)
        print(f"[✓] Saved: {dst_path}")
        saved += 1

    print(f"[✓] Done. Saved {saved} images. Skipped {skipped} due to no face detected.")
