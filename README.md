## 🧠 Image Classification using SVM (with PCA Dimensionality Reduction)

### 📌 Project Overview

This project demonstrates **image classification** using **Support Vector Machine (SVC)** combined with **Principal Component Analysis (PCA)** for dimensionality reduction.
The main goal is to classify images (e.g., handwritten letters, faces, or objects) efficiently without deep learning, while maintaining reasonable accuracy and low memory usage.

---

### 🚀 Features

* Loads and preprocesses image data
* Reduces dimensionality using **PCA**
* Trains an **SVC (Support Vector Classifier)** with optimized parameters
* Evaluates model performance with accuracy, precision, recall, and F1-score
* Includes an optional **GridSearchCV** step for hyperparameter tuning

---

### 🧩 Project Workflow

1. **Data Preparation**

   * Images are resized to `(64×64)` and converted into feature vectors.
   * Labels are encoded numerically.

2. **Feature Scaling**

   * Standardized using `StandardScaler` to normalize pixel intensity.

3. **Dimensionality Reduction**

   * `PCA` is applied to reduce the number of features from 12,288 → 200 (or other configurable values).

4. **Model Training**

   * `SVC` is trained using the RBF kernel (best found using GridSearch).

5. **Model Evaluation**

   * Performance is measured using `classification_report` and accuracy metrics.

---

### ⚙️ Optional: Hyperparameter Optimization

You can use `GridSearchCV` to automatically find the best kernel, C, and gamma values.

---

### 📊 Example Results

| Metric    | Score |
| :-------- | :---- |
| Accuracy  | 0.65  |
| Precision | 0.64  |
| Recall    | 0.66  |
| F1-score  | 0.65  |

---

### 🧾 Requirements

Install dependencies using:

```bash
pip install numpy pandas scikit-learn opencv-python tqdm matplotlib
```

---

### 🗂️ Folder Structure

```
├── data/
│   ├── train/
│   ├── test/
├── main.ipynb
├── requirements.txt
└── README.md
```

---

### 💡 Future Improvements

* Apply **HOG features** for better feature extraction
* Experiment with **CNN models (e.g., TensorFlow/Keras)**
* Try larger PCA components for improved accuracy
* Add visualizations for confusion matrix and PCA variance

