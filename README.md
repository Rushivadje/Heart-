# Heart Disease Prediction using Machine Learning

This repository contains code and data for predicting heart disease using machine learning algorithms. The goal is to use patient data to classify the likelihood of heart disease presence. The dataset, model training, evaluation, and visualizations are contained in the provided Jupyter notebook.

Table of Contents:
Project Overview
Dataset
Installation
Usage
Notebook Contents
Modeling
Evaluation
Results
License

Project Overview :

Heart disease prediction is an important application of machine learning in healthcare, allowing for proactive measures if a person is found at risk. This project applies various machine learning algorithms to classify patients with heart disease based on provided features.

Dataset:

The dataset used is heart.csv, which includes several attributes like age, sex, chest pain type, resting blood pressure, cholesterol level, maximum heart rate, etc. These features are used to predict the presence or absence of heart disease.

Features:
age: Age of the patient
sex: Sex of the patient (1 = male, 0 = female)
cp: Chest pain type
trestbps: Resting blood pressure
chol: Serum cholesterol
fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
restecg: Resting electrocardiographic results
thalach: Maximum heart rate achieved
exang: Exercise-induced angina (1 = yes; 0 = no)
oldpeak: ST depression induced by exercise
slope: Slope of the peak exercise ST segment
ca: Number of major vessels colored by fluoroscopy
thal: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
target: Heart disease (1 = presence; 0 = absence)

Installation:

To run this project, you will need Python and Jupyter Notebook installed. Additionally, install the required packages using:

Copy code
pip install -r requirements.txt
Usage

Clone this repository:
Copy code
git clone https://github.com/yourusername/heart-disease-prediction.git

Open the heart-disease-prediction.ipynb notebook in Jupyter Notebook:
Copy code
jupyter notebook heart-disease-prediction.ipynb

Run each cell sequentially to load data, preprocess, train models, and evaluate results.

Notebook Contents

The notebook is structured as follows:

Data Loading: Loading and initial inspection of the heart.csv dataset.
Data Preprocessing: Handling missing values, encoding categorical variables, scaling features.
Exploratory Data Analysis (EDA): Visualizing features to understand patterns and relationships.
Model Training: Implementing machine learning models for classification.
Evaluation: Assessing model performance using metrics like accuracy, precision, recall, and F1-score.

Modeling
Several machine learning models are used in this project, including:

Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)

Each model is trained and evaluated on the dataset to compare performance.

Evaluation
Evaluation metrics used to assess model performance:

Accuracy: Overall accuracy of predictions
Precision: True positives out of predicted positives
Recall: True positives out of actual positives
F1-Score: Harmonic mean of precision and recall
Model results are visualized using confusion matrices and ROC curves.

Results
The best model achieved an accuracy of X% (replace with actual result), showing that machine learning can be effective in predicting heart disease. Further improvements may involve feature engineering and tuning model hyperparameters.

License
This project is licensed under the MIT License. See the LICENSE file for details.

