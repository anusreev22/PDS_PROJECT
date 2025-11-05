# ------------------------------------------------------------
#  USD-INR Exchange Rate Prediction using XGBoost + Streamlit
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
#  Streamlit Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="USD-INR Prediction", layout="wide")
st.title(" USD-INR Exchange Rate Prediction")

# ------------------------------------------------------------
#  Load Dataset
# ------------------------------------------------------------
df = pd.read_csv("PDS_Project.csv")
st.subheader(" Dataset Overview")
st.dataframe(df.head())

# ------------------------------------------------------------
#  Data Cleaning & Feature Selection
# ------------------------------------------------------------
df.columns = df.columns.str.strip()

# Columns to drop
drop_cols = [
    "ECB_Deposit_Rate_Percent",
    "PBOC_Lending_Rate_Percent",
    "EU_CPI_Percent",
    "China_CPI_Percent",
    "EU_GDP_Growth_Percent",
    "EU_Unemployment_Percent",
    "EUR_INR",
    "JPY_INR",
    "CNY_INR",
    "China_GDP_Growth_Percent"
]

df = df.drop(columns=drop_cols, errors="ignore")

# Drop non-numeric and missing data
df = df.dropna()
df = df.select_dtypes(include=["float64", "int64"])

# ------------------------------------------------------------
#  Define Features and Target
# ------------------------------------------------------------
target_col = "USD_INR"
if target_col not in df.columns:
    st.error(f" '{target_col}' not found in dataset!")
    st.stop()

X = df.drop(columns=[target_col])
y = df[target_col]

# ------------------------------------------------------------
#  Train-Test Split + Model Training
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    tree_method="hist"
)

xgb_model.fit(X_train, y_train)

# ------------------------------------------------------------
#  Model Performance Metrics
# ------------------------------------------------------------
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader(" Model Performance")
st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
st.write(f"**Root Mean Square Error (RMSE):** {rmse:.4f}")
st.write(f"**R² Score:** {r2:.4f}")

# ------------------------------------------------------------
#  Default Input Values (from your screenshot)
# ------------------------------------------------------------
default_values = {
    "Brent_Crude_USD_per_Barrel": 64.061,
    "WTI_Crude_USD_per_Barrel": 60.535,
    "Equity_Flow_Rs_Crore": 1883.785,
    "Debt_Flow_Rs_Crore": 200.016,
    "Total_Flow_Rs_Crore": 2083.094,
    "Forex_Reserves_USD_Billion": 695.300,
    "Intervention_Amount_USD_Million": 0.000,
    "Exports_USD_Billion": 36.338,
    "Imports_USD_Billion": 68.103,
    "Trade_Balance_USD_Billion": -32.150,
    "RBI_Repo_Rate_Percent": 6.039,
    "Fed_Funds_Rate_Percent": 3.970,
    "India_CPI_Percent": 11.401,
    "US_CPI_Percent": 3.000,
    "India_GDP_Growth_Percent": 7.819,
    "US_GDP_Growth_Percent": 3.800,
    "India_Unemployment_Percent": 5.100,
    "US_Unemployment_Percent": 4.301,
    "Gold_Price_USD_per_oz": 3998.533
}

# ------------------------------------------------------------
#  Prediction Input Section
# ------------------------------------------------------------
st.subheader("🧮 Enter Feature Values for Prediction")

cols = st.columns(3)
input_data = {}

for i, feat in enumerate(X.columns):
    default_val = float(default_values.get(feat, df[feat].mean()))
    with cols[i % 3]:
        input_data[feat] = st.text_input(
            f"{feat}", value=f"{default_val:.3f}"
        )

# Convert inputs to DataFrame
input_df = pd.DataFrame([{k: float(v) for k, v in input_data.items()}])

# ------------------------------------------------------------
#  Predict Button
# ------------------------------------------------------------
if st.button("🔮 Predict USD-INR Rate"):
    pred = xgb_model.predict(input_df)[0]
    st.success(f"💰 **Predicted USD-INR Exchange Rate: {pred:.3f}**")

# ------------------------------------------------------------
#  Footer
# ------------------------------------------------------------
st.markdown("---")
