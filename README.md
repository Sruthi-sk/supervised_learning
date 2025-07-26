# supervised_learning
polynomial and ridge regression, feature selection, kNN generalisation, kernel perceptron, online learning

This repository contains my implementation of core supervised learning algorithms and evaluation pipelines developed as part of UCL's Supervised Learning module. All models were implemented **from scratch in NumPy**, without the use of high-level machine learning libraries like scikit-learn.

## 📁 Repository Structure

| File                                | Description                                                   |
|-------------------------------------|---------------------------------------------------------------|
| `part1_polynomial_regression.ipynb` | Polynomial regression and basis function expansion            |
| `part1_ridge_regression_boston.ipynb` | Ridge regression with feature selection on Boston dataset     |
| `part2_knn_generalization.ipynb`    | k-NN classification and generalization error experiments      |
| `kernel-perceptron_digits.ipynb`    | One-vs-All kernel perceptron on handwritten digit dataset     |
| `ovo-kernel_perceptron-digits.ipynb`| One-vs-One kernel perceptron with majority-vote prediction    |

---
## 📌 Key Topics

### 🧮 Linear & Regularized Regression
- Built least squares and ridge regression models using matrix algebra
- Applied polynomial basis functions to model non-linear relationships
- Implemented manual cross-validation to tune regularization strength (λ)
- Visualized overfitting, underfitting, and MSE trends

### 🔍 Kernel Perceptron for Digit Classification
- Implemented multi-class kernel perceptrons using polynomial and RBF kernels
- Supported both **One-vs-All (OvA)** and **One-vs-One (OvO)** strategies
- Optimized efficiency using **precomputed kernel matrices** and vectorized updates
- Performed 20-run experiments with 5-fold CV and confusion matrix analysis
- Identified ambiguous or mislabeled samples via visual inspection

### 📊 k-Nearest Neighbors & Generalization
- Simulated randomly generated binary-labeled hypotheses in 2D
- Implemented k-NN classifier and visualized decision boundaries
- Estimated generalization error across different `k` values and training sizes
- Conducted 100-run protocols to evaluate statistical stability of `k` selection

---
