# yolo_head_dataset

Dataset scaffold for training salon-specific YOLO classes.

## Classes
- `0 person`
- `1 head_customer` (customer head during haircut when full body is not visible)
- `2 staff_uniform` (staff with uniform)

## Structure
- `images/train`, `images/val`: image files (`.jpg/.jpeg/.png`)
- `labels/train`, `labels/val`: YOLO txt labels (same basename as image)
- `raw/`: source annotations before conversion
- `dataset.yaml`: Ultralytics dataset config

## Label format
Each line in `labels/*.txt`:
`<class_id> <x_center> <y_center> <width> <height>` normalized to `[0..1]`.

## Quick workflow
1. Put source annotation JSON in `raw/`.
2. Convert using `python scripts/yolo_convert_labels.py ...`.
3. Validate using `python scripts/yolo_validate_labels.py --dataset data/yolo_head_dataset`.
4. Train using `python scripts/train_yolo_head_staff.py --data data/yolo_head_dataset/dataset.yaml`.
