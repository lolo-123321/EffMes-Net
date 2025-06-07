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
Make sure you install the following dependencies:
```bash
pip install tensorflow==2.12.0
pip install keras numpy opencv-python imageio scikit-learn matplotlib seaborn dlib face_recognition
```
### Tested using:
- **OS**: Ubuntu 20.04 LTS / Google Colab
- **Hardware**: NVIDIA Tesla T4 (via Colab GPU)
- **Python**: 3.10+

## Usage Instructions
1. Choose a Notebook Based on Dataset
   - EffMes-Net_UADFV_Code.ipynb for UADFV Dataset
   - EffMes-Net_Deepfake_Code.ipynb for the YouTube Dataset
2. Upload Your Dataset to Google Drive

   -**For UADFV Dataset**:
   The dataset is automatically downloaded from Kaggle inside the notebook using the command:
   ```bash
   kaggle datasets download -d adityakeshri9234/uadfv-dataset
   ```
   
   It is then unzipped and renamed as data inside the cloned MesoNet directory.

   -**For DeepFake Dataset**:
   Download the dataset manually from:
   https://github.com/kiteco/python-youtube-code/tree/master/Deepfake-detection
   Then upload the ZIP file to your Google Drive.

   The notebook will mount your Drive and extract the dataset automatically.
4. Clone the MesoNet Base Repository
   ```bash
   git clone https://github.com/DariusAf/MesoNet.git
   cd MesoNet
   ```
5. Insert Your Model Code
   Insert the selected hybrid model into the appropriate files:
      - classifiers.py → insert classifier code.
      - example.py → insert training code.
      - predict.py → create this file and insert prediction code.
6. Run the Pipeline in Order
   ```bash
   python pipeline.py       # Preprocessing  
   python classifiers.py    # Load model  
   python example.py        # Train model  
   python predict.py        # Run prediction
   ```
## Methodology
- **Face Alignment:** Extracted using face_recognition, resized and normalized.
- **Modeling Architectures:**
     - **MesoXception:** Combines MesoNet with pretrained Xception blocks.
     - **MesoEfficient:** Combines MesoNet with EfficientNetB0 as a lightweight backbone.
- **Training Strategy:** Binary classification (Real vs. Fake), softmax activation.
- **Evaluation:**
   - Compared with standard MesoNet.
   - Plots and metrics included: accuracy, AUC, confusion matrix, ROC.

## Evaluation Metrics
- **Accuracy (ACC)** – Correct predictions over total.
- **Area Under Curve (AUC)** – Performance over varying thresholds.
- **Confusion Matrix** – Detailed classification breakdown.
- **ROC Curve** – True Positive Rate vs. False Positive Rate.

## Computing Infrastructure
- **Environment:** Google Colab
- **OS:** Linux (Ubuntu 20.04)
- **Hardware:** GPU (Tesla T4)
- **Libraries:** TensorFlow, Keras, OpenCV, Dlib, face_recognition

## Citations
 ```bash
@article{afchar2018mesonet,
  title={MesoNet: a Compact Facial Video Forgery Detection Network},
  author={Afchar, Darius and Nozick, Vincent and Yamagishi, Junichi and Echizen, Isao},
  journal={2018 IEEE International Workshop on Information Forensics and Security (WIFS)},
  pages={1--7},
  year={2018},
  organization={IEEE}
}
 ```
## License & Contribution
- Built on: Afchar's MesoNet GitHub.
- License: Research & Educational Use Only.
- Contact authors for contribution opportunities.

## Limitations
- Generalization may vary on unseen datasets outside UADFV and YouTube DeepFake.
- Face detection may fail in poor lighting or extreme angles.
- EfficientNetB0 is chosen for speed; higher variants (B1–B7) not tested.
