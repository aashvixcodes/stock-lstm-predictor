import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Forecast", layout="centered")
st.title("FutureStock: 30-Day Price Forecast")

stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, RELIANCE.NS)", "AAPL")

if st.button("Predict"):
    with st.spinner("Downloading data and predicting..."):
        df = yf.download(stock, period='6mo')
        if df.empty:
            st.error("Invalid symbol or no data found!")
        else:
            df = df[['Close']]
            df['Prediction'] = df[['Close']].shift(-30)

            X = np.array(df.drop(['Prediction'], axis=1))[:-30]
            y = np.array(df['Prediction'])[:-30]

            model = RandomForestRegressor()
            model.fit(X, y)

            future = df.drop(['Prediction'], axis=1)[-30:]
            predictions = model.predict(future)

            st.success("Prediction Complete!")
            st.subheader(f"Predicted Prices for {stock.upper()}")
            for i, price in enumerate(predictions):
                st.write(f"Day {i+1}: ₹{price:.2f}")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, 31), predictions, marker='o', linestyle='--', color='green')
            ax.set_title(f"Predicted Next 30 Days for {stock.upper()}")
            ax.set_xlabel("Future Day")
            ax.set_ylabel("Predicted Price")
            ax.grid(True)
            st.pyplot(fig)

