# 🛣️ Mask R-CNN Road Lane Detection

Welcome to the **Mask R-CNN Road Lane Detection** project!  
Detect and segment road lane markings from images using deep learning and instance segmentation.  
Let’s get started! 🚦

---

## 🧠 Project Overview

This project uses **Mask R-CNN** with a ResNet-101 backbone to find and segment different types of road lane markings, such as solid lines, dotted lines, double lines, and more.  
It’s trained on a custom COCO-format dataset for robust, multi-class lane detection.

---

## ✨ Features

- 🎯 **Instance segmentation** for road lanes
- 🏷️ Detects 7 lane marking classes + background
- 📁 **COCO format** for easy dataset management
- ⚡ Optimized training configuration
- 🖼️ Visualizes detected masks on images

---

## 🏷️ Classes Detected

- BG (Background)
- road-roads
- divider-line
- dotted-line
- double-line
- random-line
- road-sign-line
- solid-line

---

## ⚙️ Setup and Installation

1. **Clone the repository**  
   ```bash
   git clone 
   cd 
   ```

2. **Create a Python environment & install dependencies**  
   ```bash
   conda create -n mrcnn python=3.8
   conda activate mrcnn
   pip install tensorflow keras numpy matplotlib
   ```

3. **Install Mask R-CNN library**  
   - Use [Matterport’s Mask R-CNN](https://github.com/matterport/Mask_RCNN)  
   - Or for TensorFlow 2.x + Keras 2.x, use [Kamlesh364’s Mask-RCNN-TF2.7.0-keras2.7.0](https://github.com/Kamlesh364/Mask-RCNN-TF2.7.0-keras2.7.0/tree/main) 🚀
   - https://www.youtube.com/watch?v=Fu_km7FXyaU ---youtube link to setup the environment
   - In my case I used my notebook in the mrcnn environment, so that I dont have to manually install the mrcnn in the new conda environment

4. **Download COCO pretrained weights**  
   - Download `mask_rcnn_coco.h5` and place it in your project directory.

5. **Prepare your dataset**  
   - Organize your data in COCO format under the directory specified by `COCO_DIR` in the code.

---

## 🚀 Usage

### 🛠️ Configuration

Edit the `InferenceConfig` class to set:
- Number of classes
- Image size (e.g., 128x128)
- Training steps, etc.

### 🏋️‍♂️ Training

Start training your model:
```python
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=5, layers='all')
```
- Uses COCO weights for initialization
- Trains on your custom dataset

### 👀 Visualization

Display masks on images:
```python
visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
```

---

## 📁 Directory Structure

```
/logs/                        # Training logs & checkpoints
/mask_rcnn_coco.h5            # Pretrained COCO weights
/road_lane_instance_segmentation/  # Dataset (train/val annotations)
```

---

## 💡 Notes

- Images are resized to 128x128 for training 🖼️
- 8 classes (1 background + 7 lane types)
- GPU support enabled for faster training ⚡
- Logs and checkpoints saved in `logs/`

---

## 📚 References

- [Mask R-CNN by Matterport](https://github.com/matterport/Mask_RCNN)
- [Kamlesh364/Mask-RCNN-TF2.7.0-keras2.7.0](https://github.com/Kamlesh364/Mask-RCNN-TF2.7.0-keras2.7.0/tree/main) 🧠
- COCO dataset format for instance segmentation

---

## 🙌 Happy Coding!
