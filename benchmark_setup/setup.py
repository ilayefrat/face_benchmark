import os
import sys

# Add the project root to sys.path (so benchmark_setup modules can be imported)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processors.lfw_processor import process_lfw
from processors.lfw_accuracy_processor import process_lfw_accuracy
from processors.vggface_processor import process_vgg
from processors.similarity_processor import download_and_crop_images_from_excel

def confirm(prompt):
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response == 'y'

def run_setup():
    print("\n🛠️  Welcome to the Benchmark Setup Script")

    # LFW: Inversion Task
    if confirm("Do you want to process the LFW dataset for the Inversion task?"):
        lfw_root = "/home/ssd_storage/datasets/lfw_full/lfw/"
        txt_path = "./tests_datasets/inversion/lfw_test_pairs_300_upright_same.csv"
        out_dir = "./tests_data/inversion/images/"

        print(f"\nLFW source folder: {lfw_root}")
        print(f"Pairs file: {txt_path}")
        print(f"Output folders will be created in: {out_dir}")

        if confirm("Continue with this configuration?"):
            process_lfw(lfw_root, txt_path, out_dir)
        else:
            print("[✗] Skipped LFW processing.")

    # LFW: Accuracy Task
    if confirm("Do you want to process the LFW dataset for the Accuracy task?"):
        lfw_root = "/home/ssd_storage/datasets/lfw_full/lfw/"
        txt_path = "./tests_datasets/LFW/lfw_test_pairs_only_img_names.txt"

        print(f"\nLFW source folder: {lfw_root}")
        print(f"Pairs file: {txt_path}")
        print("Output folder: ./tests_data/lfw/accuracy/")

        if confirm("Continue with this configuration?"):
            process_lfw_accuracy(lfw_root, txt_path)
        else:
            print("[✗] Skipped LFW Accuracy task.")


    # VGGFace2: Other-Race Effect Task
    if confirm("Do you want to process the VGGFace2 dataset for the Other-Race task?"):
        vgg_root = "/home/ssd_storage/datasets/vggface2/train/"
        csv_asian = "./tests_datasets/other_race/vggface_other_race_same_asian.csv"
        csv_caucasian = "./tests_datasets/other_race/vggface_other_race_same_caucasian.csv"

        print(f"\nVGGFace2 source folder: {vgg_root}")
        print("CSV files to process:")
        print(f"→ Asian: {csv_asian}")
        print(f"→ Caucasian: {csv_caucasian}")
        print("Output folder: ./tests_data/other_race/stimuliLabMtcnn/")

        if confirm("Continue with this configuration?"):
            process_vgg(vgg_root, csv_asian)
            process_vgg(vgg_root, csv_caucasian)
        else:
            print("[✗] Skipped VGGFace2 processing.")

    # Similarity Task: Download and Crop Israeli and International Images
    if confirm("Do you want to process the Celebrities dataset for the Similarity tasks?"):
        excel_path = os.path.join("benchmark_setup", "img_links.xlsx")
        output_dirs = {
            "Israeli": os.path.join("tests_data", "similarity_perception", "Israeli_fam", "images"),
            "International": os.path.join("tests_data", "similarity_perception", "International_mem", "images"),
        }

        print(f"\nExcel file with image links: {excel_path}")
        for category, folder in output_dirs.items():
            print(f"→ {category}: {folder}")

        if confirm("Continue with this configuration?"):
            download_and_crop_images_from_excel(excel_path, output_dirs)
        else:
            print("[✗] Skipped Similarity task image download and cropping.")

if __name__ == "__main__":
    run_setup()
