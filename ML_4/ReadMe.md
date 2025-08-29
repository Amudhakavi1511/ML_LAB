## Aim & Objectives
To build and evaluate machine learning classifiers such as:
- Decision Tree  
- AdaBoost  
- Gradient Boosting  
- XGBoost  
- Random Forest  
- Stacked Ensemble Models (SVM, Naïve Bayes, Decision Tree, KNN)
    using **5-Fold Cross Validation** and **Hyperparameter Tuning**, and compare their performances.

---

##  Libraries Used
- **Numpy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **XGBoost**
- **ucimlrepo**

---

## Dataset
- **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Source:** UCI Machine Learning Repository  
- **Target:** `Diagnosis` → Malignant (1) / Benign (0)  
- **Preprocessing:**  
  - Removed unnecessary columns (`ID`)  
  - Encoded target variable (M → 1, B → 0)  
  - No missing values found  

---

## Exploratory Data Analysis (EDA)
- Class Distribution Plot  
- Histograms of features  
- Correlation Heatmap  
- Train-Test Split (80-20 stratified)

---

## Models Implemented
1. **Decision Tree Classifier** (GridSearchCV + Hyperparameter tuning)  
2. **AdaBoost Classifier** (Decision Stump as base learner)  
3. **Gradient Boosting Classifier**  
4. **XGBoost Classifier**  
5. **Random Forest Classifier**  
6. **Stacked Ensemble Models**  
   - SVM + Naïve Bayes + Decision Tree → Logistic Regression (meta-learner)  
   - SVM + Naïve Bayes + Decision Tree → Random Forest (meta-learner)  
   - SVM + KNN + Decision Tree → Logistic Regression (meta-learner)  

---

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Precision-Recall Curve  
- Confusion Matrix  
- Feature Importance (Model-based & Permutation Importance)  

---

## How to Run
1. Clone the repository or download the notebook.  
2. Install required dependencies:  
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost ucimlrepo
3. Run the cells step by step to:
   - Load and preprocess data
   - Train different models
   - Evaluate and compare performance
