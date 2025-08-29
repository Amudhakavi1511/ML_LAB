# Email Spam Detection  

This project implements an **Email Spam Detection** system using multiple machine learning algorithms. It demonstrates preprocessing of text data, feature engineering, model training, hyperparameter tuning, and evaluation using appropriate metrics.  

## Objective  

To classify emails as **Spam** or **Ham (Not Spam)** using supervised machine learning models and compare the performance of different algorithms.  

---

## Project Structure  
ML_3\\
|__ Classifier_models.pdf\\
|__ KNNs.ipynb\\
|__ NBs.ipynnb\\
|__ SVMs.ipynb\\

---

## Features Covered  
 
- Training multiple classifiers:  
  - **Naive Bayes** → BernoulliNB, MultinomialNB, GaussianNB  
  - **SVM** → LinearSVC, SVC (RBF, Sigmoid, Polynomial kernel)  
  - **KNN** → with BallTree, KDTree, and Brute-force search strategies  
- Hyperparameter tuning using `GridSearchCV`  
- Model comparison using:  
  - **Accuracy**  
  - **F1 Score**  
  - **ROC AUC**  
- Visualization of performance (confusion matrices, ROC curves, precision-recall curves)  

---

## Libraries Used  

- `pandas`  
- `numpy`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  
- `joblib`  

Install missing libraries using:  

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib

---

## How to run
1. Clone or download the repository.
2. Open the Jupyter notebooks in notebooks/ folder.
3. Run the cells step by step to:
   - Load and preprocess data
   - Train different models
   - Evaluate and compare performance
