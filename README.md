# Sports & Exercise Activity Recognition Using IMU Sensor Data

## 1. Introduction

- The goal is to develop an automatic activity recognition system using **IMU motion sensor data** (accelerometer & gyroscope) from the **MotionSense dataset**, which consists of multivariate time-series recordings collected from a smartphone placed in participants’ front pockets.
- The model aims to classify basic physical movements, including:
  - Walking
  - Jogging
  - Walking upstairs
  - Walking downstairs
  - Sitting
  - Standing
- By applying techniques such as **data preprocessing**, **feature extraction**, and **machine learning modeling**, this project demonstrates an **end-to-end HAR pipeline** using wearable sensor data.

---

## 2. Problem Statement

- Human Activity Recognition (HAR) using wearable sensors aims to automatically identify physical activities from inertial sensor signals.
- In this project, we use the MotionSense dataset, which contains multivariate time-series data from accelerometer and gyroscope sensors during fitness-related activities.
- The goal is to design an AI-based pipeline that processes raw IMU data and classifies human motions, addressing challenges such as sensor noise, high-dimensional time-series data, and inter-subject variability.

---

## 3. Dataset

- Primary Sensor: MotionSense (iPhone 6s – built-in IMU sensors)
- Data Format: CSV (Comma-Separated Values)
- Records: ~12,000 samples (after windowing)
- Subjects: 24
- Device: Smartphone (iPhone 6s)
- Sensors: Accelerometer, Gyroscope
- Sampling Rate: 50 Hz
- Labels: walking, jogging, sitting, standing, upstairs, downstairs
- Each record contains synchronized sensor readings including accelerometer and gyroscope signals:

|       Field             |                      Description                              |
|-------------------------|---------------------------------------------------------------|
| userAcceleration.x/y/z  | 3-axis user acceleration (m/s²), gravity removed              |
| rotationRate.x/y/z      | 3-axis gyroscope readings (rad/s)                             |
| attitude.roll           | Device roll angle (radians)                                   |
| attitude.pitch          | Device pitch angle (radians)                                  |
| attitude.yaw            | Device yaw angle (radians)                                    |
| gravity.x/y/z           | 3-axis gravity vector (m/s²)                                  |
| timestamp               | Human-readable date and time of capture                       |

## 4. Related Work
### 4.1 Activity Recognition using Smartphone and Smartwatch Sensors (WISDM)

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

### 4.2 A User-Adaptive Algorithm for Activity Recognition and Repetition Counting

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

### 4.3 Physical Activity Monitoring using the PAMAP2 Dataset

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


