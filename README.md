# Cervical-Cancer-Image-Classification

# 🧬 Cervical Cancer Image Classification with Vision Transformers

This project is a web-based image classification tool for cervical cancer cell images using three **Vision Transformer** models:

- **Swin Transformer**
- **DeiT (Data-efficient Image Transformer)**
- **CrossViT (Cross-scale Vision Transformer)**

Built using **Streamlit**, the app allows users to upload images, select models, and receive classification results in real time.

---

## 📁 Project Structure

```text
FYP_2025_Clean/
│
├── CCA/                           # Streamlit app folder
│   ├── app.py                     # Main Streamlit app
│   ├── pages/                     # Subpages for UI
│   ├── models/                    # Model definitions
│   │   └── trained_models/        # Pretrained .pth files
│   └── assets/                    # Backgrounds, icons, etc.
│
├── Train/                         # Preprocessing & training scripts
├── download_models.py             # Downloads .zip model files
├── requirements.txt               # Dependencies
├── .gitignore
└── README.md
```
---

## 🚀 How to Run

### 1. Clone the Repository

```
git clone https://github.com/dxn1405/Cervical-Cancer-Image-Classification.git
cd Cervical-Cancer-Image-Classification
```
### 2. Install Required Packages
```
pip install -r requirements.txt
```

### 3. Download the Pretrained Models
Run the script to download and extract pretrained model files automatically:

```
models.py
```

This will download:
```
Swin_best.pth
DeiT_best.pth
CrossViT_best.pth
```

All files will be placed into:
```
CCA/models/trained_models/
```
### 4. Launch the Streamlit App
```
streamlit run CCA/app.py
```

## 📊 Models Used
```
| Model        | Description                            |
| ------------ | -------------------------------------- |
| **Swin**     | Shifted Windows Transformer            |
| **DeiT**     | Data-efficient Image Transformer (ViT) |
| **CrossViT** | Cross-scale Vision Transformer         |
```

## 📥 Model File Notes
⚠️ The .pth files are not included in this repository due to GitHub's file size limit.
Instead, they are automatically downloaded from Google Drive using models.py.

If you prefer manual setup:

Upload your .pth files into 
```
CCA/models/trained_models/
```

👤 Author
---------

**Muhammad Daniel Alif Jumairi**  
Final Year Project (2025)  
*Cervical Cancer Image Classification Using Vision Transformers*
