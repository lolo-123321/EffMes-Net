# EffMes-Net

EffMes-Net is a deepfake detection system implementing two hybrid models—**MesoXception** and **MesoEfficient**—that combine MesoNet with Xception and EfficientNetB0 respectively, to achieve robust and lightweight video forgery classification.

---

## Description

This repository contains code, models, and data preprocessing scripts used for our deepfake detection research. The objective is to identify forged facial content in videos using mesoscopic and semantic features extracted from multiple CNNs. The project is built upon Afchar’s open-source MesoNet repository and extended with hybrid architectures.

---

## Dataset Information

We used two publicly available datasets:

- **UADFV Dataset**  
   [Kaggle Link](https://www.kaggle.com/datasets/adityakeshri9234/uadfv-dataset)  
  A benchmark dataset containing videos labeled as real or fake.

- **DeepFake Dataset**  
   [GitHub Link](https://github.com/kiteco/python-youtube-code/tree/master/Deepfake-detection)  
  A dataset of real/fake face videos extracted from YouTube sources.

All datasets were preprocessed using Viola-Jones face detection and Dlib facial landmark alignment.

---

## Code Structure

EffMes-Net/
├── Google_Colab_Notebooks/
│ ├── EffMes-Net_UADFV_Code.ipynb
│ └── EffMes-Net_Deepfake_Code.ipynb
├── MesoEfficient_Code/
│ ├── MesoEff_Classifier.py
│ ├── MesoEff_Training.py
│ └── MesoEff_Prediction.py
├── MesoXception_Code/
│ ├── MesoXcep_Classifier.py
│ ├── MesoXcep_Training.py
│ └── MesoXcep_Prediction.py
└── README.md

---

## Requirements

Make sure the following dependencies are installed:

```bash
pip install tensorflow==2.12.0
pip install keras numpy opencv-python imageio scikit-learn matplotlib seaborn dlib face_recognition
---
Tested using:

OS: Ubuntu 20.04 LTS / Google Colab

Hardware: NVIDIA Tesla T4 (via Colab GPU)

Python: 3.10+

---

## Usage Instructions

### 1. Choose a Notebook Based on Dataset
- `EffMes-Net_UADFV_Code.ipynb` for UADFV Dataset
- `EffMes-Net_Deepfake_Code.ipynb` for the YouTube Dataset

### 2. Upload Your Dataset to Google Drive
- Place the dataset ZIP file (`uadfv-dataset.zip` or similar) in your Drive.
- The notebooks will extract and prepare it automatically in Colab.

### 3. Clone MesoNet Base Repo
```bash
git clone https://github.com/DariusAf/MesoNet.git
cd MesoNet
