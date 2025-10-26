# 🧾 Titanic Survival Prediction using Logistic Regression

# 1. Project Title & Overview

This project focuses on building a machine learning model to predict the survival outcome of passengers aboard the RMS Titanic using historical data. Utilizing the Logistic Regression algorithm, a powerful linear model for binary classification, this analysis explores the correlation between passenger demographics and socio-economic factors (Gender, Age, Pclass, etc.) and their likelihood of survival. This serves as a foundational exercise in data cleaning, exploratory analysis, and supervised machine learning application.

# 2. Problem Statement

The sinking of the Titanic is a classic case study in disaster, where survival was not random but influenced by clear factors (often summarized as "women and children first"). The problem is to quantify and predict which passengers survived (1) or perished (0) based on their available profile data. This requires robust data preprocessing to handle missing values and feature engineering to convert categorical information into a format usable by the model.

# 3. Objectives

The primary goals of this classification project were:

Data Preparation: Load, clean, and preprocess the raw Titanic dataset to address missing values and inconsistencies.

Feature Transformation: Convert non-numeric categorical features (Sex, Embarked) into suitable numerical representations.

Model Selection: Implement and train a Logistic Regression Classifier for binary prediction.

Evaluation: Assess the model's performance using the Accuracy Score on both the training and unseen test data.

Prediction: Generate survival predictions for the test set.

# 4. Dataset Description

Attribute

Details

Source

Titanic Dataset (Kaggle/CSV source)

Samples

891 training samples (rows)

Features

Includes PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked.

Preprocessing & Cleaning

<ul><li>Missing Age: Imputed using the mean age of all passengers.</li><li>Missing Embarked: Imputed using the mode (most frequent value).</li><li>Dropping Features: The Cabin column was dropped due to a high volume of missing values. Name, Ticket, and PassengerId were also dropped as non-predictive features.</li><li>Encoding: Categorical features (Sex, Embarked) were converted to numerical format for model input.</li></ul>

Target Variable

Survived (0 = Perished, 1 = Survived)

# 5. Methodology / Approach

Data Preprocessing

Exploratory Analysis: Initial checks were performed to identify missing values (Age, Cabin, Embarked) and unique values in categorical columns.

Imputation: Missing values were handled as described above to ensure data completeness.

Feature Scaling/Standardization: The data was split into training and test sets before training to prevent data leakage.

Train-Test Split: The cleaned data was split into 80% for training and 20% for testing using train_test_split.

Model Used: Logistic Regression Classifier

Logistic Regression was chosen as the initial baseline model due to its simplicity, speed, and interpretability for binary classification problems. It models the probability of a binary outcome based on the input features.

Training, Testing, and Evaluation Strategy

Training: The Logistic Regression model was fit on the training features (X_train) and training target (Y_train).

Prediction: Predictions were generated for both the training set and the test set.

Evaluation: The accuracy score was calculated by comparing the predicted labels against the true labels for both sets, providing a measure of model fit and generalization capability.

# 6. Results & Evaluation

Performance Metrics

The model's performance was evaluated using the accuracy metric:

Training Accuracy: 80.75% (The model's performance on the data it was trained on.)

Test Accuracy: 78.77% (The model's performance on previously unseen data, indicating its generalization ability.)

Interpretation

An accuracy of approximately 79% on the test set suggests the model has learned meaningful patterns from the features (like Sex and Pclass, which are highly correlated with survival) and generalizes well to new passengers. The small gap between training and test accuracy indicates that the model is neither significantly underfitting nor overfitting the data.

# 7. Technologies Used

Category

Technology / Library

Language

Python 3.x

Data Manipulation

Pandas, NumPy

Visualization

Matplotlib, Seaborn

Modeling & Metrics

Scikit-learn (LogisticRegression, train_test_split, accuracy_score)

# 8. How to Run the Project

Prerequisites

Ensure you have a Python 3 environment installed.

# Install the necessary libraries
pip install pandas numpy matplotlib seaborn scikit-learn


Execution Guide

Obtain the standard titanic_train.csv (or similarly named) file and place it in the same directory as the notebook, or update the file path in the pd.read_csv() cell.

Save the notebook content as Titanic_Survival_Prediction.ipynb.

Open the file in a Jupyter environment (Jupyter Lab or VS Code).

Execute all cells sequentially. The final cells will output the accuracy scores for both the training and test datasets.

# 9. Conclusion

The Logistic Regression model provides a robust and interpretable baseline for predicting Titanic survival, achieving a test accuracy of nearly 79%. The project successfully navigates key machine learning steps, from handling real-world data issues (missing values) to defining an effective classification boundary. This confirms the strong predictive power of variables like Sex and Pclass on the tragic outcomes of the disaster.
