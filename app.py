import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pharmevo Forecasting System", layout="wide")

st.title("📊 Pharmevo – Sales & Inventory Forecasting")
st.markdown("**Pharma Demand Planning (6–7 Months Ahead)**")

uploaded_file = st.file_uploader("Upload Sales CSV", type=["csv"])

def calculate_order_qty(predicted_units, current_stock, safety_ratio=0.25):
    safety_stock = predicted_units * safety_ratio
    return max(0, int(predicted_units + safety_stock - current_stock))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['SalesMonth'] = pd.to_datetime(df['SalesMonth'])
    df = df.sort_values(['ProductName', 'SalesMonth'])

    st.subheader("📁 Data Preview")
    st.dataframe(df.head())

    # =============================
    # OVERALL SALES FORECAST
    # =============================
    st.subheader("📈 Overall Sales Forecast")

    overall_sales = df.groupby('SalesMonth')['TotalSales'].sum()

    sales_model = SARIMAX(
        overall_sales,
        order=(1,1,1),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    sales_result = sales_model.fit(disp=False)
    sales_forecast = sales_result.get_forecast(steps=7).predicted_mean

    fig, ax = plt.subplots()
    overall_sales.plot(ax=ax, label="Actual")
    sales_forecast.plot(ax=ax, label="Forecast")
    ax.legend()
    st.pyplot(fig)

    # =============================
    # UNITS & INVENTORY
    # =============================
    st.subheader("📦 Inventory Planning")

    overall_units = df.groupby('SalesMonth')['TotalUnits'].sum()

    units_model = SARIMAX(
        overall_units,
        order=(1,1,1),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    units_result = units_model.fit(disp=False)
    units_forecast = units_result.get_forecast(steps=7).predicted_mean

    current_stock = st.number_input("Current Warehouse Stock", value=60000)

    inventory_df = pd.DataFrame({
        "Month": units_forecast.index,
        "Predicted Units": units_forecast.values
    })

    inventory_df["Order Quantity"] = inventory_df["Predicted Units"].apply(
        lambda x: calculate_order_qty(x, current_stock)
    )

    st.dataframe(inventory_df)

    st.success("✅ Forecast & Order Plan Ready")
