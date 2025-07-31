#!/bin/bash

# ==========================
# Running All Benchmark Models
# ==========================

# 1. VGG16 | ImageNet weights | Default layers
python3 run_benchmark_id.py \
  --architecture VGG16

# 2. VGG16 | Trained on Faces | Default layers
python3 run_benchmark_id.py \
  --architecture VGG16 \
  --model-path /home/new_storage/experiments/face_memory_task/models/face_trained_vgg16_119.pth

# 3. RESNET50 | ImageNet weights | Default layers
python3 run_benchmark_id.py \
  --architecture RESNET50

# 4. IRESNET100 | ArcFace | MS1MV3 | Layer: fc
python3 run_benchmark_id.py \
  --architecture IRESNET100 \
  --model-name arcface \
  --model-path ./weights/arcface/ms1mv3_arcface_r100.pth \
  --layers-to-extract fc

# 5. IRESNET100 | CosFace | Glint360K | Layer: fc
python3 run_benchmark_id.py \
  --architecture IRESNET100 \
  --model-name cosface \
  --model-path ./weights/cosface/glint360k_cosface_r100.pth \
  --layers-to-extract fc

# 6. IRESNET100 | MagFace | MS1MV2 | Layer: fc
python3 run_benchmark_id.py \
  --architecture MAGFACE \
  --model-name magface \
  --model-path ./weights/magface/ms1mv2_magface_r100.pth \
  --layers-to-extract fc

# 7. FACENET (InceptionResNetV1) | VGGFace2 | Layer: last_bn
python3 run_benchmark_id.py \
  --architecture FACENET \
  --model-name facenet \
  --layers-to-extract last_bn

# 8. CLIP (ViT-B/32) | Pretrained | Layer: default
python3 run_benchmark_id.py \
  --architecture CLIP \
  --model-name ViT-B32

# 9. DINO (ViT-Small) | ImageNet-1K Self-supervised | Layer: default
python3 run_benchmark_id.py \
  --architecture DINO \
  --model-name vit_small_patch16_224_dino
