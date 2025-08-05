# 🎨 Polygon Colorization with Conditional U-Net

A deep learning project that performs **polygon-based colorization** of grayscale input images using a **U-Net architecture**. It conditions the output on textual input (like `"red"`, `"green"`, etc.) to color the desired regions appropriately.

---

## 📁 Project Structure

# 📌 Table of Contents

- [📌 Table of Contents](#-table-of-contents)
- [🚀 Features](#-features)
- [🧠 Model Architecture](#-model-architecture)
- [📁 Project Structure](#-project-structure)
- [🛠️ Setup Instructions](#️-setup-instructions)
- [🧪 Training](#-training)
- [🔍 Inference](#-inference)
- [📊 Experiment Tracking (WandB)](#-experiment-tracking-wandb)
- [📷 Sample Output](#-sample-output)
- [📚 References](#-references)

## 🧠 Model Highlights

- 📐 **Model:** U-Net with skip connections
- 🧾 **Input:** Grayscale image + color text prompt
- 🎯 **Output:** Colorized image
- 🧩 **Training:** MSE Loss / Cross-Entropy
- 🏁 **Goal:** Learn how color names relate to polygon regions

---
## Output :
![WhatsApp Image 2025-08-04 at 12 53 03 AM](https://github.com/user-attachments/assets/1e9c83b5-fc71-4cf5-8422-a681bffab27e)

---
## 🛠️ Setup & Installation

### 1️⃣ Create and activate virtual environment

```bash
# Create venv
python -m venv venv

# Activate venv (Windows)
venv\Scripts\activate



