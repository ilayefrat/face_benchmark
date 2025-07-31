import os
import shutil
import pandas as pd
import cv2
from benchmark_setup.utils.crop_faces import process_image

def build_filename_to_path_map(lfw_root):
    mapping = {}
    for person_folder in os.listdir(lfw_root):
        person_path = os.path.join(lfw_root, person_folder)
        if not os.path.isdir(person_path):
            continue
        for fname in os.listdir(person_path):
            full_path = os.path.join(person_path, fname)
            if fname in mapping:
                print(f"[WARNING] Duplicate file name detected: {fname}")
            mapping[fname] = full_path
    return mapping

def collect_required_images(txt_path, debug=True):
    import csv
    image_names = set()
    with open(txt_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for idx, row in enumerate(reader, start=2):
            if len(row) == 4:
                _, path1, path2, label = row
            elif len(row) == 3:
                path1, path2, label = row
            else:
                if debug:
                    print(f"[DEBUG] Skipping malformed line {idx}: {row}")
                continue
            name1 = os.path.basename(path1.strip())
            name2 = os.path.basename(path2.strip())
            image_names.update([name1, name2])

            if debug:
                print(f"[DEBUG] Line {idx}: Parsed pair {name1}, {name2}")

    if debug:
        print(f"[DEBUG] Total unique images parsed: {len(image_names)}")
    return sorted(image_names)

def process_lfw(lfw_root, txt_path, out_dir):
    print("\n[INFO] Processing LFW dataset for Inversion Task...")
    os.makedirs(out_dir, exist_ok=True)

    print("[INFO] Indexing LFW image paths...")
    filename_to_path = build_filename_to_path_map(lfw_root)

    image_names = collect_required_images(txt_path)
    print(f"[INFO] Found {len(image_names)} unique images to process.")

    for name in image_names:
        src = filename_to_path.get(name)
        if src is None or not os.path.isfile(src):
            print(f"[WARNING] Missing file: {name}")
            continue

        processed = process_image(src)
        if processed is None:
            print(f"[!] No face detected: {name}")
            continue

        upright_path = os.path.join(out_dir, name)
        inverted_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".flipped.jpg")

        cv2.imwrite(upright_path, processed)
        flipped = cv2.rotate(processed, cv2.ROTATE_180)
        cv2.imwrite(inverted_path, flipped)

        print(f"[✓] Saved: {name} and {os.path.basename(inverted_path)}")

    print("[✓] Done processing LFW for inversion task.")
