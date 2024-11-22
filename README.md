# Breast Cancer Prediction using Machine Learning

This project aims to predict breast cancer diagnosis (malignant or benign) using various machine learning algorithms. The primary goal is to compare the performance of different models and identify the most effective approach for accurate prediction.

## Project Overview

Breast cancer is one of the most common types of cancer worldwide, and early detection is critical for improving survival rates. In this project, I applied ten machine learning models to predict whether a given breast mass is benign or malignant based on input features. The models analyzed include:

- **K Nearest Neighbors (KNN)**
- **Support Vector Machines (SVM)**
- **Logistic Regression**
- **Multiple Linear Regression**
- **Ridge Regression**
- **Decision Trees**
- **Random Forest**
- **Naive Bayes**
- **Bagging**
- **K Means Clustering**

The dataset used is the **Wisconsin Breast Cancer Dataset (WBCD)**, which contains information about cell features from breast cancer biopsies.

## Models Used

### 1. K Nearest Neighbors (KNN)
- A non-parametric method used for classification and regression.
- Achieved **99.12% accuracy** in detecting breast cancer.

### 2. Support Vector Machine (SVM)
- A supervised learning algorithm for classification tasks.
- High accuracy but with a higher computational cost.

### 3. Logistic Regression
- A statistical model used for binary classification.
- A simple model that provided competitive results in this project.

### 4. Multiple Linear Regression
- A linear approach for modeling the relationship between a dependent variable and multiple independent variables.

### 5. Ridge Regression
- A type of linear regression that uses L2 regularization to reduce overfitting.

### 6. Decision Trees
- A non-linear model used for classification that splits the dataset into branches based on feature values.

### 7. Random Forest
- An ensemble method that combines multiple decision trees to improve accuracy and prevent overfitting.

### 8. Naive Bayes
- A probabilistic classifier based on Bayes' theorem, often used for classification problems with categorical features.

### 9. Bagging
- A method that improves the stability and accuracy of machine learning algorithms by reducing variance and overfitting.

### 10. K Means Clustering
- An unsupervised learning algorithm used for clustering data based on feature similarity.

## Key Metrics

- **Accuracy**
- **F1-score**
- **Precision**
- **Recall**

These metrics are used to evaluate the models' performance and determine the most reliable model for breast cancer detection.

## Dataset

The dataset used in this project is the **Wisconsin Breast Cancer Dataset** available from the UCI Machine Learning Repository. It contains 30 numerical attributes (such as radius, texture, smoothness, etc.) and a label indicating whether the tumor is malignant (M) or benign (B).

## Requirements

- Python 3.x
- Libraries:  
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib (for visualization)
  - Seaborn (for visualization)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aashita-radhika/Breast-Cancer-Machine-Learning-Project.git
