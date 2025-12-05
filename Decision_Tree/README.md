# Decision Tree Classifier (Iris Dataset)

This module demonstrates a simple Decision Tree classification model applied to the classic Iris dataset. It includes model training, evaluation, and visualizations such as a confusion matrix, feature importance chart, and the rendered decision tree structure.

## Overview

The Decision Tree algorithm is a supervised learning method used for classification. It splits the data based on feature values to make predictions. Decision Trees are easy to interpret and perform well on small to medium datasets.

### What this project includes

- `decision_tree.py`  
  Main script that loads data, trains the model, evaluates performance, and generates plots.

- `confusion_matrix.png`  
  Visualization showing how well the model classified each iris species.

- `feature_imp.png`  
  Bar chart displaying feature importance scores.

- `tree.png`  
  A visual representation of the full Decision Tree structure.

## Dataset

The Iris dataset contains 150 samples of iris flowers with 4 input features:
- Sepal length  
- Sepal width  
- Petal length  
- Petal width  

Target classes:
- Setosa  
- Versicolor
The script will:
   - Train the Decision Tree
   - Print accuracy and classification report
   - Generate and save 3 images in this directory

## Model Evaluation

The project includes:

### 1. Confusion Matrix
Shows correct vs incorrect predictions.

### 2. Classification Report
Includes:
- Precision  
- Recall  
- F1-score  

### 3. Feature Importance
Indicates which features influence decisions the most.

### 4. Decision Tree Plot
A full visualization of the trained model and splits.

## Next Steps (Optional Improvements)

- Hyperparameter tuning (`max_depth`, `criterion`, `min_samples_split`)
- Train/test cross-validation
- Add Random Forest and compare performance
- Export model using `joblib` or `pickle`

---

This project is intended as a lightweight, readable example of classic machine learning model workflows in Python.
- Virginica  

