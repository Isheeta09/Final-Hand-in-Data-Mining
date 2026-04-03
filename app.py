import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Customer_Turnover per month.xlsx")
    return df

data = load_data()

# -----------------------------
# FEATURE ENGINEERING (same as notebook)
# -----------------------------
customer_revenue = data.groupby("Debiteur").agg({
    "Omzet": "sum",
    "Periode": "count"
}).reset_index()

customer_revenue.columns = ["Debiteur", "total_revenue", "active_months"]

customer_revenue["avg_monthly_revenue"] = (
    customer_revenue["total_revenue"] / customer_revenue["active_months"]
)

# -----------------------------
# CHURN CREATION
# -----------------------------
last_purchase = data.groupby("Debiteur").agg({
    "Jaar": "max",
    "Periode": "max"
}).reset_index()

last_purchase["month_index"] = last_purchase["Jaar"] * 12 + last_purchase["Periode"]

latest_month = last_purchase["month_index"].max()

last_purchase["months_since_last_purchase"] = (
    latest_month - last_purchase["month_index"]
)

last_purchase["churn"] = (
    last_purchase["months_since_last_purchase"] >= 12
).astype(int)

# -----------------------------
# MERGE DATA
# -----------------------------
model_data = customer_revenue.merge(
    last_purchase[["Debiteur", "churn"]],
    on="Debiteur",
    how="left"
)

# -----------------------------
# MODEL TRAINING
# -----------------------------
X = model_data[["total_revenue", "active_months", "avg_monthly_revenue"]]
y = model_data["churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_scaled, y)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Customer Churn Prediction App")

st.write("Enter customer information to predict churn risk:")

total_revenue = st.number_input("Total Revenue", min_value=0.0, value=10000.0)
active_months = st.number_input("Active Months", min_value=1, value=12)
avg_monthly_revenue = st.number_input("Average Monthly Revenue", min_value=0.0, value=800.0)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Churn"):
    
    input_data = np.array([[total_revenue, active_months, avg_monthly_revenue]])
    input_scaled = scaler.transform(input_data)
    
    prediction = knn_model.predict(input_scaled)[0]
    probability = knn_model.predict_proba(input_scaled)[0][1]
    
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error("Customer is likely to CHURN")
    else:
        st.success("Customer is likely to STAY")
    
    st.write(f"Churn Probability: {round(probability, 2)}")
    
    # Risk segmentation
    if probability > 0.7:
        st.write("Risk Level: HIGH")
    elif probability > 0.4:
        st.write("Risk Level: MEDIUM")
    else:
        st.write("Risk Level: LOW")