import os
import argparse
import requests
import pandas as pd
from tqdm import tqdm

def download_images_from_excel(excel_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_excel(excel_file)
    required_cols = {"url", "name", "category", "is_original"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Excel file must contain the following columns: {required_cols}")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        url = row["url"]
        name = row["name"]
        category = row["category"]
        
        # Ensure file extension
        if not any(name.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
            name += ".jpg"
        
        category_folder = os.path.join(output_dir, category)
        os.makedirs(category_folder, exist_ok=True)
        filepath = os.path.join(category_folder, name)

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/122.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from an Excel file.")
    parser.add_argument("--excel", required=True, help="Path to the Excel file with image metadata.")
    parser.add_argument("--outdir", required=True, help="Directory to save the images.")

    args = parser.parse_args()
    download_images_from_excel(args.excel, args.outdir)

''' Run this script with the following command:
python download_images.py --excel img_links.xlsx --outdir downloaded_images/
'''