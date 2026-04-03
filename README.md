# Data Mining Project (DATDRD05-HC-T06)
This project predicts customer churn, estimates revenue at risk, and performs customer segmentation using machine learning for a B2B company.

## Project Overview

The objective of this project is to:
- Predict customer churn  
- Estimate revenue at risk  
- Segment customers based on risk  
- Calculate short-term Customer Lifetime Value (CLV)
  
## Dataset

The dataset consists of ERP data from 2023–2025 including:
- Customer transactions  
- Revenue  
- Activity patterns  

## Data Preparation

- Data cleaning and preprocessing  
- Feature engineering:
  - Total revenue  
  - Active months  
  - Average monthly revenue  
- Creation of churn label  

## Machine Learning Models

Three models were developed:

- k-Nearest Neighbors (kNN)  
- Naive Bayes  
- Logistic Regression  

## Model Evaluation
Models were evaluated using:
- Accuracy  
- Recall  

## Customer Risk Segmentation
Customers were grouped into:
- Low Risk  
- Medium Risk  
- High Risk  

This helps prioritize retention strategies.

## Revenue at Risk
Revenue was aggregated across risk segments to estimate financial impact.

## Customer Lifetime Value (CLV)
Short-term CLV was calculated to identify high-value customers at risk.

## Streamlit Application
An interactive Streamlit app was built to:
- Input customer data  
- Predict churn in real-time  

## How to Run the App

1. Download the repository  
2. Open terminal in project folder  
3. Install dependencies:
   pip install streamlit pandas scikit-learn  
4. Run:
   streamlit run app.py  

## Conclusion
This project demonstrates how machine learning can be applied to solve real business problems by combining prediction with business insights.
