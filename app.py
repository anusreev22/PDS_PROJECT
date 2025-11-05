# ------------------------------------------------------------
#  USD-INR Exchange Rate Prediction using ML Models + Streamlit
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ------------------------------------------------------------
#  Streamlit Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="USD-INR Prediction", layout="wide")
st.title("💱 USD-INR Exchange Rate Prediction Dashboard")

# ------------------------------------------------------------
#  Load Dataset
# ------------------------------------------------------------
df = pd.read_csv("PDS_Project.csv")
st.subheader("📊 Dataset Overview")
st.dataframe(df.head())

# ------------------------------------------------------------
#  Data Cleaning & Feature Selection
# ------------------------------------------------------------
df.columns = df.columns.str.strip()

drop_cols = [
    "ECB_Deposit_Rate_Percent", "PBOC_Lending_Rate_Percent",
    "EU_CPI_Percent", "China_CPI_Percent", "EU_GDP_Growth_Percent",
    "EU_Unemployment_Percent", "EUR_INR", "JPY_INR",
    "CNY_INR", "China_GDP_Growth_Percent"
]
df = df.drop(columns=drop_cols, errors="ignore")
df = df.dropna()
df = df.select_dtypes(include=["float64", "int64"])

# ------------------------------------------------------------
#  Define Features and Target
# ------------------------------------------------------------
target_col = "USD_INR"
if target_col not in df.columns:
    st.error(f"'{target_col}' not found in dataset!")
    st.stop()

X = df.drop(columns=[target_col])
y = df[target_col]

# ------------------------------------------------------------
#  Train-Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
#  Model Selection
# ------------------------------------------------------------
st.subheader("⚙️ Select Model for Prediction")

model_choice = st.radio(
    "Choose a regression model:",
    ("XGBoost Regressor", "Random Forest Regressor", "Linear Regression")
)

if model_choice == "XGBoost Regressor":
    model = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1, random_state=42, tree_method="hist"
    )
elif model_choice == "Random Forest Regressor":
    model = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42
    )
else:
    model = LinearRegression()

# ------------------------------------------------------------
#  Train Selected Model
# ------------------------------------------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------------------------------------------------------------
#  Visualization Tabs
# ------------------------------------------------------------
st.subheader("📉 Visualization")

tab1, tab2, tab3 = st.tabs(["Predicted vs Actual", "Residual Plot", "Feature Importance"])

with tab1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual USD-INR")
    ax.set_ylabel("Predicted USD-INR")
    ax.set_title("Predicted vs Actual USD-INR")
    st.pyplot(fig)

with tab2:
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(residuals, bins=20, kde=True, ax=ax)
    ax.set_title("Residual Distribution")
    st.pyplot(fig)

with tab3:
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=fi.values[:10], y=fi.index[:10], ax=ax)
        ax.set_title("Top 10 Important Features")
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for Linear Regression.")

# ------------------------------------------------------------
#  Prediction Input Section
# ------------------------------------------------------------
st.subheader("🧮 Enter Feature Values for Prediction")

default_values = {
    "Brent_Crude_USD_per_Barrel": 64.061, "WTI_Crude_USD_per_Barrel": 60.535,
    "Equity_Flow_Rs_Crore": 1883.785, "Debt_Flow_Rs_Crore": 200.016,
    "Total_Flow_Rs_Crore": 2083.094, "Forex_Reserves_USD_Billion": 695.300,
    "Intervention_Amount_USD_Million": 0.000, "Exports_USD_Billion": 36.338,
    "Imports_USD_Billion": 68.103, "Trade_Balance_USD_Billion": -32.150,
    "RBI_Repo_Rate_Percent": 6.039, "Fed_Funds_Rate_Percent": 3.970,
    "India_CPI_Percent": 11.401, "US_CPI_Percent": 3.000,
    "India_GDP_Growth_Percent": 7.819, "US_GDP_Growth_Percent": 3.800,
    "India_Unemployment_Percent": 5.100, "US_Unemployment_Percent": 4.301,
    "Gold_Price_USD_per_oz": 3998.533
}

cols = st.columns(3)
input_data = {}
for i, feat in enumerate(X.columns):
    default_val = float(default_values.get(feat, df[feat].mean()))
    with cols[i % 3]:
        input_data[feat] = st.number_input(f"{feat}", value=default_val)

input_df = pd.DataFrame([input_data])

# ------------------------------------------------------------
#  Prediction Button
# ------------------------------------------------------------
if st.button("🔮 Predict USD-INR Rate"):
    pred = model.predict(input_df)[0]
    st.success(f"💰 **Predicted USD-INR Exchange Rate ({model_choice}): {pred:.3f}**")

# ------------------------------------------------------------
#  Save Model (optional)
# ------------------------------------------------------------
if st.button("💾 Save Trained Model"):
    filename = f"{model_choice.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)
    st.info(f"Model saved successfully as **{filename}**")

# ------------------------------------------------------------
#  Footer
# ------------------------------------------------------------
st.markdown("---")

