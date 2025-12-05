# decision_tree_iris.py
# Basic Decision Tree classifier on the Iris dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # 1. Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    target_names = iris.target_names

    # Optional: put into a DataFrame (nice for EDA)
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    print("First 5 rows of the dataset:")
    print(df.head(), "\n")

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Define and train the Decision Tree model
    model = DecisionTreeClassifier(
        criterion="gini",      # or "entropy"
        max_depth=3,           # limit depth so tree is readable
        random_state=42
    )
    model.fit(X_train, y_train)

    # 4. Predictions
    y_pred = model.predict(X_test)

    # 5. Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 6. Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title("Confusion Matrix - Decision Tree (Iris)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

    # 7. Feature importance plot
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(7, 4))
    sns.barplot(
        x=importances[indices],
        y=np.array(feature_names)[indices]
    )
    plt.title("Feature Importances - Decision Tree (Iris)")
    plt.xlabel("Importance score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # 8. Visualize the tree itself
    plt.figure(figsize=(12, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=target_names,
        filled=True,
        rounded=True
    )
    plt.title("Decision Tree Structure - Iris Dataset")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
