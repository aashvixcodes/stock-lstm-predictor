import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Stock Forecast", layout="centered")
st.title("FutureStock (LSTM): 30-Day Price Forecast")

stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, RELIANCE.NS)", "AAPL")

if st.button("Predict"):
    with st.spinner("Training LSTM and predicting future prices..."):
        # 1. Download 5 years of historical stock data
        df = yf.download(stock, period='5y')
        if df.empty:
            st.error("Invalid symbol or no data found!")
        else:
            # 2. Use only 'Close' column and scale it
            data = df[['Close']].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # 3. Create sequences: 60 previous days to predict next day
            X, y = [], []
            for i in range(60, len(scaled_data) - 30):
                X.append(scaled_data[i-60:i, 0])
                y.append(scaled_data[i:i+30, 0])  # 30-day forecast

            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # reshape for LSTM

            # 4. Build LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model.add(LSTM(50))
            model.add(Dense(30))  # Predict 30 future prices
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=10, batch_size=64, verbose=0)

            # 5. Predict the next 30 days using last 60 days
            last_60_days = scaled_data[-60:]
            input_seq = last_60_days.reshape(1, 60, 1)
            predicted_scaled = model.predict(input_seq)
            predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

            # 6. Display and plot
            st.success("Prediction Complete!")
            st.subheader(f"Predicted Prices for {stock.upper()} (Next 30 Days)")
            for i, price in enumerate(predicted):
                st.write(f"Day {i+1}: ₹{price[0]:.2f}")

            # Plot prediction
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, 31), predicted, marker='o', linestyle='--', color='blue')
            ax.set_title(f"Predicted Next 30 Days for {stock.upper()} (LSTM)")
            ax.set_xlabel("Future Day")
            ax.set_ylabel("Predicted Price")
            ax.grid(True)
            st.pyplot(fig)
