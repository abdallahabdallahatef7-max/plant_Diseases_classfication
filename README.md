
# 🌿 AgriVision: Technical Documentation & Code Overview

This repository contains an end-to-end plant disease classification system. Below is a detailed technical breakdown of the implementation.

## 🧠 Code Architecture Breakdown

### 1. Model Structure (`PlantDiseaseClassifier`)

The model uses **EfficientNet-B0** as a feature extractor.

* **Backbone:** We utilize `models.efficientnet_b0(weights='DEFAULT')`.
* **Feature Extraction:** The original classifier head is replaced with `nn.Identity()`. This allows us to extract the high-level 1280-dimensional feature vector from the image.
* **Custom Classifier (MLP):**
* **Layer 1:** Fully connected layer (1280 to 512 units) + ReLU + Dropout (0.4).
* **Layer 2:** Fully connected layer (512 to 128 units) + ReLU + Dropout (0.3).
* **Layer 3 (Output):** 128 units to 38 classes (Logits).



### 2. Training Pipeline Logic

The training script includes several advanced features to ensure model stability:

* **Preprocessing:** Images are resized to $224 \times 224$ and normalized using ImageNet statistics: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
* **Early Stopping:** A custom monitor tracks `Validation Loss`. If the loss doesn't improve for **5 consecutive epochs**, training terminates to prevent overfitting.
* **Model Checkpointing:** Only the version of the model with the lowest validation loss is saved as `best_plant_model.pth`.

### 3. Backend Implementation (FastAPI)

The production server is built using **FastAPI** for high-performance asynchronous handling.

* **Image Processing:** Uploaded images are converted to RGB using `Pillow`, transformed into tensors, and passed to the model.
* **Inference:** The model runs in `eval()` mode with `torch.no_grad()` to speed up calculations and reduce memory usage.
* **Response:** Returns the predicted disease name and the confidence percentage calculated via a **Softmax** function.

### 4. Frontend & Deployment

* **UI/UX:** A modern interface using CSS variables for a "Nature-inspired" theme.
* **Dockerization:** The environment is isolated using a `python:3.10-slim` base image to ensure the app runs identically on any server.

---

## 🛠️ Setup and Installation

### Local Development

1. **Clone the repo:**
```bash
git clone https://github.com/your-username/your-repo-name.git

```


2. **Install requirements:**
```bash
pip install -r requirements.txt

```


3. **Run Server:**
```bash
uvicorn main:app --reload

```



### Deployment via Docker

```bash
docker build -t plant-disease-detector .
docker run -p 8000:8000 plant-disease-detector

```

---

## 📋 Classes Supported (38 Total)

The model classifies diseases across various species including:

* **Apple:** Scab, Black Rot, Rust, Healthy.
* **Corn:** Common Rust, Gray Leaf Spot, Northern Blight, Healthy.
* **Tomato:** Early Blight, Late Blight, Yellow Leaf Curl, Mosaic Virus, etc.
* **Potato, Grape, Peach, Pepper, and more.**

---
# Finally link my app : https://huggingface.co/spaces/abdullah-ml/plant_diseases_classification0
