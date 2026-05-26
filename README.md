# Machine Learning Models README

## Introduction

This repository contains an overview of three important supervised machine learning algorithms:

1. Linear Regression
2. Logistic Regression
3. Support Vector Machine (SVM)

These algorithms are widely used for prediction, classification, and data analysis tasks in machine learning.

---

# 1. Linear Regression

## Overview

Linear Regression is a statistical and machine learning technique used to predict continuous numerical values by finding relationships between features and labels.

It attempts to fit the best straight line through the dataset.

### Linear Regression Equation

[
y' = b + w_1x_1
]

Where:

* (y') = Predicted output
* (b) = Bias (intercept)
* (w_1) = Weight (slope)
* (x_1) = Input feature

For multiple features:

[
y' = b + w_1x_1 + w_2x_2 + ... + w_nx_n
]

## Key Concepts

* Predicts continuous values
* Uses weights and bias
* Training adjusts model parameters to minimize loss
* Gradient Descent is commonly used for optimization
* Works well when relationships are linear

## Applications

* House price prediction
* Sales forecasting
* Weather prediction
* Stock trend analysis

## Advantages

* Simple and easy to understand
* Fast training process
* Interpretable results

## Limitations

* Performs poorly with non-linear data
* Sensitive to outliers
* Assumes linear relationship

## Training Process

1. Initialize weights and bias
2. Calculate predictions
3. Compute loss
4. Use Gradient Descent to reduce loss
5. Repeat until convergence

---

# 2. Logistic Regression

## Overview

Logistic Regression is a supervised learning algorithm used for classification problems.

Unlike Linear Regression, Logistic Regression predicts probabilities instead of continuous values.

It is commonly used for binary classification tasks.

Examples:

* Spam or Not Spam
* Pass or Fail
* Disease Detection

## Sigmoid Function

Logistic Regression uses the sigmoid function to map outputs between 0 and 1.

[
f(x) = \frac{1}{1 + e^{-x}}
]

The model first calculates:

[
z = b + w_1x_1 + w_2x_2 + ... + w_nx_n
]

Then applies the sigmoid function:

[
y' = \frac{1}{1 + e^{-z}}
]

## Key Concepts

* Used for classification tasks
* Outputs probabilities
* Uses Sigmoid Function
* Uses Log Loss instead of Squared Loss
* Regularization helps prevent overfitting

## Applications

* Email spam detection
* Fraud detection
* Medical diagnosis
* Customer churn prediction

## Advantages

* Efficient and simple
* Works well for binary classification
* Probabilistic interpretation

## Limitations

* Not suitable for highly complex relationships
* Can underperform on non-linear datasets

## Loss Function

Logistic Regression uses Log Loss:

[
\text{Log Loss} = -\frac{1}{N}\sum_{i=1}^{N} y_i\log(y_i') + (1-y_i)\log(1-y_i')
]

Regularization methods:

* L2 Regularization
* Early Stopping

---

# 3. Support Vector Machine (SVM)

## Overview

Support Vector Machine (SVM) is a supervised machine learning algorithm used for both classification and regression tasks.

SVM works by finding the optimal hyperplane that separates data points into different classes.

The goal is to maximize the margin between classes.

## Key Concepts

* Hyperplane
* Support Vectors
* Margin Maximization
* Kernel Functions

## Types of Kernels

### Linear Kernel

Used when data is linearly separable.

### Polynomial Kernel

Used for curved decision boundaries.

### Radial Basis Function (RBF)

Most commonly used kernel for non-linear problems.

### Sigmoid Kernel

Behaves similarly to neural networks.

## Applications

* Image classification
* Face detection
* Text categorization
* Bioinformatics
* Handwriting recognition

## Advantages

* Effective in high-dimensional spaces
* Works well with clear margin separation
* Memory efficient

## Limitations

* Slow for very large datasets
* Choosing the correct kernel can be difficult
* Less effective with noisy data

## SVM Workflow

1. Select kernel function
2. Map data into higher dimensions
3. Find optimal hyperplane
4. Maximize margin between classes
5. Predict class labels

---

# Comparison Table

| Feature       | Linear Regression | Logistic Regression   | SVM                       |
| ------------- | ----------------- | --------------------- | ------------------------- |
| Problem Type  | Regression        | Classification        | Classification/Regression |
| Output        | Continuous Value  | Probability/Class     | Class Label               |
| Main Function | Linear Equation   | Sigmoid Function      | Hyperplane                |
| Loss Function | Squared Loss      | Log Loss              | Hinge Loss                |
| Best For      | Linear data       | Binary classification | High-dimensional data     |

---

# Conclusion

Linear Regression, Logistic Regression, and Support Vector Machines are foundational machine learning algorithms.

* Linear Regression is best for predicting continuous numerical values.
* Logistic Regression is widely used for classification and probability prediction.
* SVM is powerful for complex classification tasks with clear margins.

Understanding these algorithms is essential for building strong machine learning models.

---

# References

* Google Machine Learning Crash Course
* Developers Google Machine Learning Documentation
* Machine Learning Fundamentals

