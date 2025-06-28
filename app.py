import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Forecast", layout="centered")
st.title("📈 30-Day Stock Price Predictor")

stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, RELIANCE.NS)", "AAPL")

if st.button("Predict"):
    with st.spinner("🔄 Downloading data and training model..."):
        # Download historical data
        df = yf.download(stock, period='5y')

        if df.empty:
            st.error("⚠️ Invalid symbol or no data found!")
        else:
            df = df[['Close']]
            data = df.values

            # Normalize data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            # Prepare training sequences
            X, y = [], []
            for i in range(60, len(scaled_data)):
                X.append(scaled_data[i-60:i, 0])
                y.append(scaled_data[i, 0])
            X = np.array(X)
            y = np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=5, batch_size=32, verbose=0)

            # Forecast next 30 days
            future_input = scaled_data[-60:].reshape(1, 60, 1)
            predictions = []
            for _ in range(30):
                next_price = model.predict(future_input, verbose=0)[0][0]
                predictions.append(next_price)
                future_input = np.append(future_input[:, 1:, :], [[[next_price]]], axis=1)
            predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

            # Display results
            st.success("✅ Prediction Complete!")
            st.subheader(f"Predicted Prices for {stock.upper()} (Next 30 Days)")
            for i, price in enumerate(predicted_prices):
                st.write(f"Day {i+1}: ₹{price[0]:.2f}")

            # Plot graph
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, 31), predicted_prices, marker='o', linestyle='--', color='green')
            ax.set_title(f"{stock.upper()} - 30 Day Forecast")
            ax.set_xlabel("Day")
            ax.set_ylabel("Predicted Close Price")
            ax.grid(True)
            st.pyplot(fig)

