**#Diabetes Prediction using Binary Classification (Machine Learning)**

## Problem Statement
This project focuses on building a machine learning classification model to predict diabetes based on patient’s medical attributes. The objective is to analyze key factors involved to diagnose diabetes in an individual.

## Problem Context
Diabetes is a major healthcare concern worldwide. Early prediction enables preventive care, timely diagnosis, and better treatment outcomes. This project demonstrates how data-driven models can support clinical decision-making.

## Dataset
The dataset contains diagnostic health measurements from a publicly available dataset.
Source: Kaggle – Pima Indians Diabetes Dataset
Input Features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- Body Mass Index (BMI)
- Diabetes Pedigree Function
- Age
Target Variable:
- Outcome (0 = Non - Diabetic, 1 = Diabetic)

**[Dataset Features] (images./Dataset_features_diabetes.png)**

## Notebook (Google Colab)
This project notebook can be executed directly in Google Colab without any local setup.
[Open in Google Colab] https://colab.research.google.com/drive/1r_wWsz3Q8Y12vikfxS2FUR4PPOtSfRTj?usp=sharing

## Technical Approach
- Data cleaning and visualization
- Feature engineering
- Model training and comparison  
- Model evaluation using classification metrics
- Usage of Binary Classification as supervised learning model where the target variable has exactly two possible classes. 

## Tech Stack
Python, Pandas, Scikit-learn, Seaborn, Matplotlib

## Machine Learning Models
- Logistic Regression  
- Decision Tree
- Random Forest Classifier  
- K-Nearest Neighbors (KNN) 
- Naive Bayes Model 
- Ensemble Methods (Gradient Boosting and Ada Boost Model)

## Model Evaluation
Evaluation was performed using the following metrics:
- Precision  
- Accuracy  
- Recall  
- F1-Score  
- ROC-AUC Score 
 
## Results
- Achieved accuracy of 76.62% on test data.
- Logistic Regression achieved a ROC AUC of 0.73.

  **[ROC Curve] (images./ROC_curve_diabetes_dataset.png)**

- Random Forest Classifier demonstrated superior predictive performance compared to other models.

  **[Performance Scores]:**

  **{Before Preprocessing} (images./Performance_scores_before_preprocessing.png)**

  **{After Preprocessing} (images./Performance_scores_after_preprocessing.png)**
 
## Project Structure
diabetes-prediction/
├── data/
│ └── diabetes.csv # Dataset
├── notebooks/
│ └── diabetes_prediction.ipynb # Exploratory analysis & modeling
├── src/
├── preprocessing.py   # Data cleaning, scaling, encoding
├── train.py           # Model training logic
└── evaluate.py        # Metrics, ROC AUC, plots
├── requirements.txt # Dependencies
└── README.md # Project documentation

## Setup & Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction.git
   cd diabetes-prediction
2. Create and activate a virtual environment to isolate project dependencies.
    python -m venv venv
    source venv/bin/activate    #macOS/Linux
    venv\Scripts\activate         # Windows
3. Install dependencies:
    pip install -r requirements.txt
4. Run the notebook:
    jupyter notebook
5. (Recommended) Run on Google Colab by uploading the notebook/clicking on ‘Open on Colab’ button.

## Key Learnings
- Gained insights into the importance of data scaling, as model performance varied significantly before and after preprocessing.
- Utilized correlation matrices to examine relationships among features.
  
  **[Correlation Matrix] (images./Heatmap_diabetes_dataset.png)**

  **[Correlation Matrix] (images./Pairplot_diabetes_dataset.png)**

- Acquired practical experience processing and analyzing real-world healthcare data.

## Disclaimer
This project is developed for educational purposes using a publicly available dataset and is not intended for clinical or medical diagnosis.

