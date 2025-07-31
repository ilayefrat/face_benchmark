import os
import pandas as pd
import cv2
from benchmark_setup.utils.crop_faces import process_image

def parse_vggface2_path(filename):
    """
    F_A_n006750_0015_01.jpg → folder: n006750, image: 0015_01.jpg
    """
    parts = filename.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected filename format: {filename}")
    folder = parts[2]  # 'n006750'
    image_name = "_".join(parts[3:])  # '0015_01.jpg'
    return folder, image_name

def copy_and_process_vgg(csv_path, vgg_root, output_base, debug=True):
    df = pd.read_csv(csv_path)
    os.makedirs(output_base, exist_ok=True)

    seen = set()
    for idx, row in df.iterrows():
        for col in ['img1', 'img2']:
            fname = row[col]
            if fname in seen:
                if debug:
                    print(f"[DEBUG] Skipping duplicate: {fname}")
                continue
            seen.add(fname)

            try:
                folder, img_name = parse_vggface2_path(fname)
                src_path = os.path.join(vgg_root, folder, img_name)
                dst_path = os.path.join(output_base, fname)
            except Exception as e:
                print(f"[!] Error parsing {fname}: {e}")
                continue

            if not os.path.isfile(src_path):
                print(f"[WARNING] Missing file: {src_path}")
                continue

            processed = process_image(src_path)
            if processed is None:
                print(f"[!] No face detected: {fname}")
                continue

            cv2.imwrite(dst_path, processed)
            print(f"[✓] Saved: {dst_path}")

    print(f"[INFO] Total unique images processed: {len(seen)}")

def process_vgg(vgg_root, csv_path):
    print("\n[INFO] Processing VGGFace2 dataset for Other-Race Task...")
    output_base = "./tests_data/other_race/images/"

    print(f"[INFO] Reading CSV file: {csv_path}")
    print(f"[INFO] VGGFace2 root folder: {vgg_root}")
    print(f"[INFO] Output folder: {output_base}")

    copy_and_process_vgg(csv_path, vgg_root, output_base)
    print("[✓] Done processing VGGFace2.")
