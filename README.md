# EffMes-Net

EffMes-Net is a deepfake detection system implementing two hybrid models—MesoXception and MesoEfficient—that combine MesoNet with Xception and EfficientNetB0 respectively, to achieve robust and lightweight video forgery classification.

## Description
This repository contains code, models, and data preprocessing scripts used for our deepfake detection research. The objective is to identify forged facial content in videos using mesoscopic and semantic features extracted from multiple CNNs.

## Dataset Information
We used two publicly available datasets:
- **UADFV Dataset** – [Kaggle Link](https://www.kaggle.com/datasets/adityakeshri9234/uadfv-dataset)
- **DeepFake Dataset** – [GitHub Link](https://github.com/kiteco/python-youtube-code/tree/master/Deepfake-detection)

Each dataset includes real and manipulated face videos. Face regions were extracted using Viola-Jones and Dlib landmarks.

## Code Information
The code includes:
- CNN model definitions (`MesoXception`, `MesoEfficient`)
- Data preprocessing (face extraction, alignment)
- Training and evaluation scripts
- Utilities for metrics and visualization (ROC, confusion matrix)

## Usage Instructions
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Preprocess the dataset**:
   - Extract and align faces using `preprocess.py`

3. **Train the model**:
   ```bash
   python train.py --model mesoXception
   ```

4. **Evaluate the model**:
   ```bash
   python evaluate.py --model mesoXception
   ```

## Requirements
- Python 3.8+
- TensorFlow / Keras
- OpenCV
- Dlib
- face_recognition

## Methodology
- **Preprocessing**: Face detection and alignment using Viola-Jones and Dlib.
- **Model**: Hybrid models combining MesoNet with either Xception or EfficientNetB0.
- **Training**: Augmentation, dropout regularization, ADAM optimizer.
- **Evaluation**: Comparative performance on UADFV and DeepFake datasets.

## Computing Infrastructure
- OS: Google Colab (Linux backend)
- GPU: Tesla T4
- RAM: ~13GB

## Evaluation Method
Comparative evaluation with:
- Meso-4, MesoInception-4
- Xception-raw, Xception-c40
- Capsule Networks
- Multi-task networks

## Assessment Metrics
- **Accuracy (ACC)**: Measures correct classification rate.
- **AUC (Area Under ROC Curve)**: Measures model’s ability to separate real vs fake.

## Limitations
- Limited real-time applicability due to preprocessing overhead.
- Performance may drop on unseen deepfake generation techniques.

## Citation
If using this code or models, please cite our manuscript:
> [EffMes-Net: A Novel Deep Learning-Based Approach for Deepfake Detection Utilizing Meso-Net](DOI placeholder)

## License
This project is open-sourced under the MIT License.

## Contribution Guidelines
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you’d like to change.

---
This README fulfills PeerJ Computer Science's reproducibility and code sharing guidelines for AI applications.
