# 📌 Image Generation & Processing with Stable Diffusion and Flux (Julia)

This project focuses on generating synthetic images using **Stable Diffusion**, preprocessing them for analysis, and demonstrating a **CNN-based feature extraction model** using **Flux.jl**. The workflow includes **three main tasks**:  

1. **Task 1:** Generating synthetic images with Stable Diffusion  
2. **Task 2:** Preprocessing images (resizing, grayscale conversion)  
3. **Task 3:** Running a CNN model in Flux.jl to process the images  

---

## 🚀 Task 1: Image Generation with Stable Diffusion  

We use **Stable Diffusion v1.4** to generate synthetic images based on text prompts. The model is loaded using the **diffusers** library, and **autocast** is used to optimize performance on CUDA GPUs.

### ✅ **Approach**  
- Used **pretrained Stable Diffusion** to generate **3 images** based on a prompt.  
- Applied **autocast()** for efficient computation on GPUs.  
- The generated images were **saved locally** for further processing.  

### ⚠️ **Challenges & Assumptions**  
- **Model requires a GPU** (running on CPU is significantly slower).  
- Output quality depends on **prompt engineering**.  
- **Generated images may vary** due to the probabilistic nature of diffusion models.  

---

## 🖼️ Task 2: Image Preprocessing  

We preprocess the generated images to make them suitable for model input using **PyTorch and OpenCV**.

### ✅ **Approach**  
- Resized images to **224x224** (standard input for CNNs).  
- Converted images to **grayscale**.  
- Applied **normalization** to scale pixel values to **[-1,1]**.  
- Saved processed images for further use.  

### ⚠️ **Challenges & Assumptions**  
- **Maintaining aspect ratio** was not prioritized (fixed 224x224 size).  
- Assumed all images are **RGB**; grayscale conversion was optional.  
- Required correct **file paths** to avoid errors in batch processing.  

---

## 🏗️ Task 3: CNN-Based Feature Extraction with Flux.jl  

A **Convolutional Neural Network (CNN)** was implemented using **Flux.jl** to process the preprocessed images.

### ✅ **Approach**  
- Used a **simple CNN model** with **Conv2D, MaxPooling, and Dense layers**.  
- Computed the correct **flattened input size** before the dense layer.  
- Loaded **preprocessed grayscale images** as input tensors.  
- Ran the model **forward pass** and observed feature extraction results.  

### ⚠️ **Challenges & Assumptions**  
- **Input image dimensions must be correct** (expected 224x224).  
- Model trained with **randomly initialized weights** (not pre-trained).  
- **Normalization was required** to match the CNN's expected input range.

## 🛠️ Installation & Setup  

### **1️⃣ Clone the repository**  
```sh
git clone https://github.com/your-username/image-gen-processing.git
cd image-gen-processing
```

### **2️⃣ Install Dependencies**  

#### **🔹 For Python (Image Generation & Preprocessing)**  
```sh
pip install torch torchvision diffusers transformers opencv-python numpy matplotlib
```

#### **🔹 For Julia (CNN Model with Flux.jl)**  
```sh
julia -e 'using Pkg; Pkg.add(["Flux", "Images", "ImageIO"])'
```

---


