# 5G User Prediction Project

## Project Background

During the 2022 World Internet Conference Wuzhen Summit, the "World Internet Development Report 2022" indicated that in the first quarter of 2022, the number of global 5G users increased by 70 million, reaching about 620 million, with a 5G population coverage exceeding 25%. It is predicted that by the end of 2022, the number of global 5G users will exceed 1 billion. Telecom operators need to create user profiles based on user-side information and further target potential 5G users for precise marketing.

## Project Goal

Utilize basic user information and communication-related data (such as billing information, data usage, active behavior, plan type, region information, etc.) to train models to predict whether each user is a 5G user.

## Data Description

The dataset contains 800,000 entries, each with 60 fields:

- `id`: Sample identifier
- `target`: Prediction target
- `cat`: 20 categorical features
- `num`: 38 numerical features

## Model Selection and Analysis

### 1. Logistic Regression

- **Accuracy**: 0.9864
- **AUC**: 0.5002
- **Analysis**: The model has high accuracy but a low AUC value, with low recall and F1 scores for class 1.0. The model fails to identify samples of class 1.0 due to data imbalance. Sample imbalance needs to be addressed.

### 2. Random Forest

- **Accuracy**: 0.9867
- **AUC**: 0.9181
- **Analysis**: The model performs best, but improvements are still needed to handle data imbalance.

### 3. Decision Tree

- **Accuracy**: 0.98637
- **AUC**: 0.8818
- **Analysis**: The model has high prediction accuracy but a relatively low AUC value. Adjusting parameters can improve model performance.

### 4. Linear Discriminant Analysis (LDA)

- **Accuracy**: 0.9834
- **AUC**: 0.8527
- **Analysis**: The model has high accuracy but still needs to address data imbalance.

### 5. K-Nearest Neighbors (KNN)

- **Accuracy**: 0.9866
- **AUC**: 0.8344
- **Analysis**: The model has good performance and can be improved by adjusting parameters.

### 6. Gaussian Naive Bayes

- **Accuracy**: 0.3633
- **AUC**: 0.7513
- **Analysis**: The model performs poorly and needs further optimization.

## Conclusion and Next Steps

- **Summary**: The Random Forest model performs the best in this experiment, but all models have poor performance in identifying minority class samples due to data imbalance.
- **Next Steps**: Address sample imbalance issues (e.g., oversampling, undersampling) and conduct more in-depth feature engineering (e.g., feature selection, feature scaling) to improve model performance.

## Project Structure

```
5G-User-Prediction/

├── DecisionTree.py  # Decision Tree script
├── GaussianNaiveBayes.py  # Gaussian Naive Bayes script
├── KNN.py  # K-Nearest Neighbors script
├── LDA.py  # Linear Discriminant Analysis script
├── LogisticRegression.py    # Logistic Regression script
├── RandomForest.py    # Random Forest script
├── README.md            # Project description
└── 人工智能导论.ipynb     # jupyter notebook
```

## Installation and Running

Clone the project

```
git clone https://github.com/Qiaokeliyeye/Forecast-of-5G-Users.git
cd Forecast-of-5G-Users
```

## License

This project is licensed under the MIT License. See LICENSE for more details.