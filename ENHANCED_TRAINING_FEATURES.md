# Enhanced Training Features for HGCameraCounter

This document describes the enhanced training and evaluation features added to the HGCameraCounter project.

## New Training Features

### 1. Improved Early Stopping
- **Patience Control**: Configure how many epochs to wait before stopping when no improvement is seen
- **Minimum Delta**: Set the minimum change in metrics required to qualify as improvement

### 2. Checkpoint Management
- **Checkpoint Period**: Save model checkpoints every N epochs to prevent losing progress
- **Automatic Recovery**: Continue training from the last saved checkpoint if training is interrupted

### 3. Advanced Optimizer Parameters
- **Learning Rate Control**: Configure initial and final learning rates
- **Momentum Settings**: Adjust SGD momentum parameters
- **Weight Decay**: Control optimizer weight decay
- **Warmup Parameters**: Configure warmup epochs and related settings

### 4. Automatic Hyperparameter Optimization
- **Bayesian Optimization**: Use evolutionary algorithms to find optimal hyperparameters
- **Configurable Iterations**: Set how many optimization iterations to run
- **Automatic Application**: Best parameters are automatically applied to final training

## New Evaluation Features

### 1. Verbose Output
- Detailed metrics display during evaluation
- Progress information and timing statistics

### 2. Per-Class Metrics
- Class-specific precision, recall, and mAP scores
- Detailed breakdown for each object class in your dataset

## UI Enhancements

### Model Train Tab
New controls added:
- Patience (Early Stopping)
- Checkpoint Period
- Hyperparameter Optimization checkbox
- Optimization Iterations spinner

### Model Test Tab
New controls added:
- Verbose Output checkbox
- Per-Class Metrics checkbox

### Dataset Lab Tab
Enhanced with:
- Automatic verbose and per-class evaluation
- Detailed logging of all metrics in the output pane

## Usage Examples

### Basic Training with Enhanced Features
```bash
python scripts/train_yolo_head_staff.py \
  --data data/yolo_head_dataset/dataset.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --patience 20 \
  --checkpoint-period 5 \
  --lr0 0.01 \
  --lrf 0.001
```

### Training with Hyperparameter Optimization
```bash
python scripts/train_yolo_head_staff.py \
  --data data/yolo_head_dataset/dataset.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --hyperparameter-opt \
  --opt-iters 100
```

### Evaluation with Detailed Metrics
```bash
python scripts/eval_yolo_head_staff.py \
  --weights runs/train/yolo_head_staff/weights/best.pt \
  --data data/yolo_head_dataset/dataset.yaml \
  --verbose \
  --per-class
```

## Benefits

1. **Better Model Performance**: Hyperparameter optimization can significantly improve model accuracy
2. **Safer Training**: Checkpoint saving prevents loss of training progress
3. **More Control**: Fine-grained control over training parameters
4. **Detailed Analysis**: Per-class metrics help identify which classes need more training
5. **Time Savings**: Early stopping prevents unnecessary training epochs

These enhancements make the HGCameraCounter system more professional and suitable for production deployment.