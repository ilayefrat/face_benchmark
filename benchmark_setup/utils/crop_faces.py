import os
import sys
import cv2
import numpy as np
from mtcnn import MTCNN
import mediapipe as mp

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
detector = MTCNN()
TARGET_SIZE = (112, 112)  # You can change to (160, 160) if needed

def align_face(image, left_eye, right_eye):
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    center = (int(image.shape[1] // 2), int(image.shape[0] // 2))
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def crop_tight_face(image, box, scale=0.6):
    x, y, w, h = box
    img_h, img_w = image.shape[:2]

    # Expand the box by scale
    x_center = x + w / 2
    y_center = y + h / 2
    new_w = w * (1 + scale)
    new_h = h * (1 + scale)

    x1 = int(max(x_center - new_w / 2, 0))
    y1 = int(max(y_center - new_h / 2, 0))
    x2 = int(min(x_center + new_w / 2, img_w))
    y2 = int(min(y_center + new_h / 2, img_h))

    return image[y1:y2, x1:x2]

def apply_segmentation(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = segmenter.process(rgb)
    mask = result.segmentation_mask

    # Threshold the segmentation mask
    binary_mask = mask > 0.5

    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

    # Apply mask
    output = np.zeros_like(image)
    output[clean_mask.astype(bool)] = image[clean_mask.astype(bool)]
    return output

def center_square_crop(image):
    h, w = image.shape[:2]
    size = min(h, w)
    x = (w - size) // 2
    y = (h - size) // 2
    return image[y:y+size, x:x+size]

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    # 1. Detect face FIRST on the original image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(rgb)
    if not result:
        return None

    face = result[0]
    box = face['box']
    keypoints = face['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # 2. Apply segmentation (on original image, BEFORE alignment or crop)
    segmented = apply_segmentation(image)

    # 3. Align the segmented image using eyes from detection
    aligned = align_face(segmented, left_eye, right_eye)

    # 4. OPTIONAL: Re-detect on aligned image (not mandatory)
    result_aligned = detector.detect_faces(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    if result_aligned:
        aligned_box = result_aligned[0]['box']
    else:
        aligned_box = box  # fallback to original box if no detection

    # 5. Crop using the aligned box
    cropped = crop_tight_face(aligned, aligned_box, scale=0.6)

    # 6. Center square crop
    square_crop = center_square_crop(cropped)

    # 7. Resize
    resized = cv2.resize(square_crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized

def process_folder(folder):
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for fname in os.listdir(subfolder_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            fpath = os.path.join(subfolder_path, fname)
            try:
                processed = process_image(fpath)
                if processed is not None:
                    # Decide saving format based on folder name
                    if "Israeli" in subfolder:
                        # Save as PNG
                        save_path = os.path.splitext(fpath)[0] + ".png"
                        cv2.imwrite(save_path, processed)
                        # Optionally: remove the old file if extension changed
                        if not fpath.endswith('.png'):
                            os.remove(fpath)
                        print(f"Processed (saved as PNG): {save_path}")
                    else:
                        # Save as JPG
                        save_path = os.path.splitext(fpath)[0] + ".jpg"
                        cv2.imwrite(save_path, processed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        if not fpath.endswith('.jpg') and not fpath.endswith('.jpeg'):
                            os.remove(fpath)
                        print(f"Processed (saved as JPG): {save_path}")
                else:
                    print(f"No face detected: {fpath}")
            except Exception as e:
                print(f"Error processing {fpath}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python crop_faces_final.py <input_folder>")
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a valid directory.")
        sys.exit(1)

    process_folder(input_dir)
