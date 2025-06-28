import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Streamlit App Layout
st.set_page_config(page_title="Stock LSTM Predictor", layout="centered")
st.title("📈 30-Day Stock Price Forecast")

# User Input
stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY.NS)", "AAPL")

if st.button("Predict"):
    try:
        with st.spinner("Downloading data and training the model..."):
            df = yf.download(stock, period="5y")
            if df.empty:
                st.error("❌ No data found for the given symbol.")
            else:
                df = df[['Close']]
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df)

                # Prepare training data
                X, y = [], []
                for i in range(60, len(scaled_data)):
                    X.append(scaled_data[i-60:i, 0])
                    y.append(scaled_data[i, 0])
                X = np.array(X)
                y = np.array(y)
                X = X.reshape((X.shape[0], X.shape[1], 1))

                # Build the LSTM Model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                    LSTM(50),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X, y, epochs=5, batch_size=32, verbose=0)

                # Forecast Next 30 Days
                input_seq = scaled_data[-60:].reshape(1, 60, 1)
                predictions = []
                for _ in range(30):
                    pred = model.predict(input_seq, verbose=0)[0][0]
                    predictions.append(pred)
                    input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

                predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

                # Output
                st.success("✅ Prediction Complete!")
                st.subheader(f"📊 Predicted Closing Prices for {stock.upper()} (Next 30 Days)")
                for i, price in enumerate(predicted_prices):
                    st.write(f"Day {i+1}: ₹{price[0]:.2f}")

                # Plotting
                fig, ax = plt.subplots()
                ax.plot(range(1, 31), predicted_prices, marker='o', color='green', linestyle='--')
                ax.set_xlabel("Day")
                ax.set_ylabel("Price")
                ax.set_title(f"{stock.upper()} - 30 Day Forecast")
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
