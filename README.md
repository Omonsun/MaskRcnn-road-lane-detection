# Mask R-CNN Road Lane Detection

This repository contains an implementation of a road lane detection system using Mask R-CNN, a deep learning framework for instance segmentation. The model is trained on a custom dataset (COCO format) to detect multiple classes of road lane markings.

## Project Overview

The project leverages the Mask R-CNN architecture with a ResNet-101 backbone to perform instance segmentation on road lane images. It detects and segments various types of lane markings such as solid lines, dotted lines, double lines, road signs, and dividers.

## Features

- Instance segmentation of road lanes with Mask R-CNN
- Supports 7 lane marking classes plus background
- Uses COCO dataset format for training and validation
- Training configuration optimized for lane detection
- Visualization of detected masks on images

## Classes Detected

- Background (BG)
- road-roads
- divider-line
- dotted-line
- double-line
- random-line
- road-sign-line
- solid-line

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone 
   cd 
   ```

2. **Create a Python environment and install dependencies:**
   ```bash
   conda create -n mrcnn python=3.8
   conda activate mrcnn
   pip install tensorflow keras numpy matplotlib
   ```

3. **Install the Mask R-CNN library dependencies:**
   - Clone the [Mask R-CNN repository by Matterport](https://github.com/matterport/Mask_RCNN) or use [Kamlesh364's Mask R-CNN for TensorFlow 2.7.0 and Keras 2.7.0](https://github.com/Kamlesh364/Mask-RCNN-TF2.7.0-keras2.7.0/tree/main) for easy TF2 compatibility.

4. **Download the COCO pretrained weights:**
   - Download `mask_rcnn_coco.h5` from the official Mask R-CNN repository and place it in the project directory.

5. **Prepare your dataset in COCO format** under the directory specified by `COCO_DIR` in the code.

## Usage

### Configuration

Modify the `InferenceConfig` class to adjust parameters such as number of classes, image size, and training steps.

### Training

Run the training script to start training the model:

```python
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=5, layers='all')
```

- Training data and validation data are loaded from the COCO dataset directory.
- The model is initialized with COCO weights and fine-tuned on the road lane dataset.

### Visualization

Visualize masks on sample images using:

```python
visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
```

## Directory Structure

```
/logs/             # Directory for saving training logs and checkpoints
/mask_rcnn_coco.h5 # Pretrained COCO weights
/road_lane_instance_segmentation/ # Dataset directory with train and val annotations
```

## Notes

- The model uses images resized to 128x128 for training.
- The number of classes is set to 8 (1 background + 7 lane classes).
- GPU support is enabled for training.
- Training logs and checkpoints are saved under the `logs` directory.

## References

- [Mask R-CNN by Matterport](https://github.com/matterport/Mask_RCNN)
- [Kamlesh364/Mask-RCNN-TF2.7.0-keras2.7.0 (TensorFlow 2.x + Keras 2.x compatible Mask R-CNN)](https://github.com/Kamlesh364/Mask-RCNN-TF2.7.0-keras2.7.0/tree/main)
- COCO dataset format for instance segmentation
