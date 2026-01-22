# Sports & Exercise Activity Recognition Using IMU Sensor Data

## 1. Introduction

This repository contains the final project for the course **Human Activity Recognition & Machine Learning**.  
The goal is to develop an automatic activity recognition system using **IMU motion sensor data** (accelerometer & gyroscope), capable of classifying and counting physical activities such as walking, running, squatting, and strength exercises.

This work simulates features found in smart wearable devices (e.g., smartwatches) for **automatic exercise detection** and **repetition counting**, supporting the concept of **Quantified Self** — leveraging personal data to better understand and improve physical activity habits.

---

## 2. Problem Statement

Human activity recognition (HAR) from wearable sensors plays an important role in health monitoring, sports analytics, and personalized fitness tracking.

The system analyzes time-series signals from 3-axis accelerometer and gyroscope sensors to automatically distinguish between different labeled activities using a trained machine learning model.

Key objectives:

1. Collect and preprocess sensor time-series data.
2. Train a classification model to recognize activity types.
3. Evaluate model performance on held-out test data.
4. Support activity counting for repetitive movement exercises.

---

## 3. Dataset

### 3.1 Data Source  
- **Primary Sensor:** MetaMotion Sensor  
- **Data Format:** CSV (Comma-Separated Values)

Each record includes synchronized sensor readings with:

| Field            | Description                                                |
|------------------|------------------------------------------------------------|
| `epoch (ms)`     | Timestamp in milliseconds for synchronization              |
| `time`           | Human-readable date and time of capture                    |
| `elapsed (s)`    | Duration (seconds) since the start of recording            |
| `acc_x/y/z (g)`  | 3-axis accelerometer readings (G-force units)              |
| `gyr_x/y/z (deg/s)`| 3-axis gyroscope readings (degrees per second)          |

### 3.2 Activities & Participants  
- **Tracked Activities:**  
  - Bench press  
  - Deadlift  
  - Overhead press  
  - Squat  
  - Walking  
  - Running

- **Participants:**  
  - Data collected from **5 volunteers** following a structured strength training program.
## 4. Related Work

This project is inspired by and built upon several influential studies in the fields of **Human Activity Recognition (HAR)**, **fitness analytics**, and **wearable sensor–based machine learning**. The following works provide important methodological and empirical foundations.

---

**Paper:** *Intelligent Fitness Data Analysis and Training Effect Prediction Based on Machine Learning Algorithms*

**Models & Approach**
- Evaluates multiple supervised learning models for activity classification:
  - **Random Forest (RF):** Main model, using ensemble decision trees to improve robustness and reduce overfitting.
  - **Support Vector Machine (SVM):** Used to construct optimal decision boundaries between activities.
  - **Naive Bayes (NB):** Serves as a lightweight baseline for speed comparison.
- Workflow follows a classical pipeline of **signal preprocessing, feature extraction, and classification**, similar to this project.

**Dataset**: [link](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)
- Collected from wearable devices (smartwatches / IMU sensors).
- Time-series data including **3-axis accelerometer** and **heart rate signals**.
- Activities include running, cycling, squats, and push-ups.

**Key Findings**
- Random Forest achieved the highest accuracy (>90%).
- RF demonstrated strong robustness under noisy sensor conditions.
- The study confirms the feasibility of predicting training effectiveness using ML-based activity recognition.

---

### 4.2 Activity Recognition using Smartphone and Smartwatch Sensors (WISDM)  
**Authors:** Kwapisz et al.  
**Source:** ACM SIGKDD Explorations / IEEE Access / UCI WISDM Dataset

**Models & Approach**
- Proposes a structured HAR pipeline:
  - Fixed-length **time-window segmentation**
  - Noise reduction and outlier handling
  - Extraction of over 40 statistical and domain-specific features
  - **PCA** for dimensionality reduction
- Classification using:
  - Naive Bayes
  - Random Forest
  - Support Vector Machine

**Dataset**: [link](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
- **WISDM Dataset**
- 51 participants, 18 activities
- Dual-device setup:
  - Smartphone (pocket)
  - Smartwatch (dominant hand)
- Sensors: tri-axial accelerometer and gyroscope

**Key Findings**
- Random Forest achieved the highest overall accuracy (>90%).
- Smartwatch data is more effective for upper-body activities.
- Smartphone data performs better for locomotion tasks.
- Sensor fusion significantly improves recognition performance.

---

### 4.3 A User-Adaptive Algorithm for Activity Recognition and Repetition Counting

**Models & Approach**
- Introduces a **user-adaptive HAR framework**:
  - **K-Means clustering** for initializing user-specific activity patterns
  - **Local Outlier Factor (LOF)** for removing unreliable samples
  - **Multivariate Gaussian models** for activity intensity classification
- Includes probability-based fall detection.

**Dataset**: [link](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)
- IMU sensor data (accelerometer-based)
- Activities grouped by intensity: light, moderate, vigorous
- Multi-user dataset for adaptability evaluation

**Key Findings**
- User-adaptive learning improves recognition accuracy.
- Reduces manual labeling requirements.
- Suitable for real-time wearable and healthcare applications due to low computational cost.

---

### 4.4 Physical Activity Monitoring using the PAMAP2 Dataset

**Models & Approach**
- Preprocessing: interpolation, noise filtering, outlier removal
- Feature extraction from both time and frequency domains (FFT, signal energy)
- **PCA** applied for dimensionality reduction
- Classification using Naive Bayes, SVM, and Random Forest

**Dataset**: [link](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
- **PAMAP2 Dataset (UCI)**
- 9 participants
- 3 IMUs (wrist, chest, ankle) + heart rate monitor
- 18 activities, 52 attributes per timestamp

**Key Findings**
- Random Forest achieved the best performance (>90% accuracy).
- Naive Bayes was computationally efficient but less accurate.
- Multi-sensor fusion significantly improved recognition accuracy.
- PCA enabled near real-time inference.

---

### 4.5 RecGym: Gym Workouts Recognition Dataset  
**Authors:** Ceriani et al.  
**Source:** Sensors (MDPI), 2020

**Models & Approach**
- Signal smoothing using low-pass and moving average filters
- Feature extraction:
  - Signal Magnitude Area (SMA)
  - Peak-to-peak values
  - Time-domain statistics
- PCA for dimensionality reduction
- Classification using Random Forest (primary) and SVM
- Repetition counting via peak detection and zero-crossing analysis

**Dataset**: [link](https://archive.ics.uci.edu/dataset/1128/recgym:+gym+workouts+recognition+dataset+with+imu+and+capacitive+sensor-7)
- **RecGym Dataset**
- 10 participants
- Exercises: Squat, Bench Press, Arm Curl, Leg Curl, etc.
- Sensors: IMU + capacitive sensors

**Key Findings**
- Random Forest achieved F1-score >94%
- Repetition counting error <3%
- Multimodal sensor fusion outperformed IMU-only approaches

---

## 5. Model Selection and Motivation

To ensure a fair and comprehensive evaluation, multiple machine learning models were implemented in this project. Each model serves a specific purpose and provides insight into different aspects of the feature space and data characteristics.

---

### 5.1 Naive Bayes (NB) – Probabilistic Baseline

**Reason for Use:**  
Naive Bayes is a simple probabilistic classifier with very few hyperparameters. It is primarily used as a baseline model to verify whether the extracted features contain meaningful information.

**Role in the Project:**  
- Acts as a reference point for performance comparison  
- Helps identify whether strong feature dependencies exist  
- Poor performance indicates complex, non-independent feature relationships

---

### 5.2 K-Nearest Neighbors (KNN) – Distance-Based Classification

**Reason for Use:**  
KNN is included to evaluate the discriminative power of the feature space based on distance metrics rather than learned parameters.

**Role in the Project:**  
- Checks whether activities form well-separated clusters in feature space  
- Evaluates feature quality without assuming a parametric model  
- Sensitive to feature scaling and noise, providing insight into preprocessing effectiveness

---

### 5.3 Decision Tree (DT) – Interpretable Decision Rules

**Reason for Use:**  
Decision Trees provide an interpretable model capable of capturing non-linear decision boundaries. Their rule-based structure makes them easy to understand and visualize.

**Role in the Project:**  
- Helps interpret which features are most influential  
- Serves as a comparison point for Random Forest to analyze overfitting  
- Provides transparency for model behavior analysis

---

### 5.4 Random Forest (RF) – Robust Ensemble Model

**Reason for Use:**  
Random Forest is an ensemble method that aggregates multiple decision trees to improve generalization and reduce variance.

**Role in the Project:**  
- Serves as the **primary model** due to its robustness and strong performance  
- Handles noisy sensor data effectively  
- Consistently achieves the best accuracy across experiments

---

### 5.5 Neural Network (MLP) – General Nonlinear Modeling

**Reason for Use:**  
A Multi-Layer Perceptron (MLP) is used to evaluate whether a general non-linear model can outperform traditional tree-based approaches.

**Role in the Project:**  
- Tests the benefit of learned non-linear representations  
- Compares deep learning capability against classical ML models  
- Evaluates whether model complexity yields performance gains on this dataset

---

### 5.6 Why Multiple Models Are Used

Using multiple machine learning models is essential for a reliable and unbiased evaluation.

- Different models exhibit **different inductive biases**, meaning they learn different types of relationships from the same data.
- According to the **No Free Lunch Theorem**, no single model is optimal for all problems.
- Comparing multiple models helps:
  - Avoid biased conclusions  
  - Evaluate the stability and robustness of the feature set  
  - Select the model best suited to the underlying data distribution  


