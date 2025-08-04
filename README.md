# 🎨 Polygon Colorization with Conditional U-Net

A deep learning project that performs **polygon-based colorization** of grayscale input images using a **U-Net architecture**. It conditions the output on textual input (like `"red"`, `"green"`, etc.) to color the desired regions appropriately.

---

## 📁 Project Structure


---

## 🧠 Model Highlights

- 📐 **Model:** U-Net with skip connections
- 🧾 **Input:** Grayscale image + color text prompt
- 🎯 **Output:** Colorized image
- 🧩 **Training:** MSE Loss / Cross-Entropy
- 🏁 **Goal:** Learn how color names relate to polygon regions

---

## 🛠️ Setup & Installation

### 1️⃣ Create and activate virtual environment

```bash
# Create venv
python -m venv venv

# Activate venv (Windows)
venv\Scripts\activate


