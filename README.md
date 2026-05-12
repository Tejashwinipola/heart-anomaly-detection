Heart Anomaly Detection – Cardiac Sound Classification using CNN
A deep learning system that classifies heart sounds as normal or abnormal using a Convolutional Neural Network (CNN). The model processes raw audio signals through signal processing and feature extraction pipelines to enable early detection of cardiac anomalies.

Table of Contents

Overview
Problem Statement
Approach
Tech Stack
Project Structure
Model Architecture
Results
Getting Started
Future Improvements


Overview
Heart disease is one of the leading causes of death globally. Early detection through non-invasive cardiac sound analysis can significantly improve patient outcomes. This project builds an automated pipeline that takes raw heart sound audio, processes it, extracts meaningful features, and classifies it using a CNN — making it suitable for early anomaly screening.

Problem Statement
Manually analyzing phonocardiogram (PCG) signals requires trained cardiologists and is time-consuming. The goal of this project is to automate the classification of heart sounds into:

Normal — healthy cardiac rhythm
Abnormal — irregular sounds indicating potential cardiac conditions


Approach
Raw Audio (.wav)
      ↓
Butterworth Bandpass Filter (noise removal)
      ↓
Mel-Spectrogram Feature Extraction
      ↓
Data Augmentation (handle class imbalance)
      ↓
CNN Model Training
      ↓
Binary Classification: Normal / Abnormal
1. Signal Preprocessing

Applied Butterworth bandpass filter to remove low-frequency noise and high-frequency artifacts from raw PCG signals
Filtered frequency range: 25 Hz – 400 Hz (standard cardiac sound range)

2. Feature Extraction

Converted filtered audio to Mel-spectrograms — 2D visual representations of sound frequency over time
Mel-spectrograms capture both time and frequency characteristics, making them ideal input for CNN image classifiers

3. Data Augmentation

Dataset had class imbalance (more normal samples than abnormal)
Applied techniques: time stretching, pitch shifting, adding background noise
Balanced the dataset to prevent model bias toward the majority class

4. CNN Classification

Treated Mel-spectrograms as images and trained a CNN to learn discriminative patterns
Used convolutional layers to extract spatial features from the spectrogram


Tech Stack
ComponentTechnologyLanguagePython 3.xDeep LearningTensorFlow / KerasSignal Processinglibrosa, scipyData HandlingNumPy, pandasVisualizationmatplotlib, seabornNotebook EnvironmentJupyter Notebook / Google Colab

Project Structure
heart-anomaly-detection/
├── data/
│   ├── normal/              # Normal heart sound samples (.wav)
│   └── abnormal/            # Abnormal heart sound samples (.wav)
│
├── notebooks/
│   └── heart_anomaly_detection.ipynb   # Full pipeline notebook
│
├── src/
│   ├── preprocess.py        # Butterworth filter + Mel-spectrogram extraction
│   ├── augment.py           # Data augmentation functions
│   ├── model.py             # CNN architecture definition
│   └── train.py             # Model training script
│
├── models/
│   └── cnn_model.h5         # Saved trained model
│
├── results/
│   ├── confusion_matrix.png
│   └── training_history.png
│
└── README.md

Model Architecture
Input: Mel-Spectrogram (128 x 128 x 1)
      ↓
Conv2D (32 filters, 3x3) + ReLU + MaxPooling
      ↓
Conv2D (64 filters, 3x3) + ReLU + MaxPooling
      ↓
Conv2D (128 filters, 3x3) + ReLU + MaxPooling
      ↓
Flatten
      ↓
Dense (256) + ReLU + Dropout (0.5)
      ↓
Dense (1) + Sigmoid
      ↓
Output: Normal (0) / Abnormal (1)

Results
MetricScore
Accuracy 90%
Precision 92%
Recall 94%
F1 Score 91%


Getting Started
Prerequisites
Python 3.8+
TensorFlow 2.x
librosa
scipy
numpy
pandas
matplotlib
Installation
bash# Clone the repository
git clone https://github.com/Tejashwinipola/heart-anomaly-detection.git
cd heart-anomaly-detection

# Install dependencies
pip install -r requirements.txt
Or open directly in Google Colab by uploading the notebook file.
Dataset
This project uses the PhysioNet/CinC Challenge 2016 heart sound dataset.

Download: https://physionet.org/content/challenge-2016/1.0.0/
Place normal samples in data/normal/ and abnormal samples in data/abnormal/


Future Improvements

Multi-class classification (identify specific cardiac conditions)
Real-time prediction via a web or mobile interface
Deploy as a REST API using Flask or FastAPI
Experiment with transfer learning (VGG16, ResNet on spectrograms)
Larger and more diverse dataset for better generalization


Author
Pola Tejashwini

LinkedIn: linkedin.com/in/tejashwini-pola-a5b3b1280
Email: ptejashwini7@gmail.com


License
This project is licensed under the MIT License.
