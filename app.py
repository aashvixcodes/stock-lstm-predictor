import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import os

# Set page title
st.set_page_config(page_title="Stock Price Predictor")

# Cache data fetching to avoid repeated downloads
@st.cache_data
def fetch_data(ticker, period="1y"):
    df = yf.download(ticker, period=period)
    return df

# Preprocess data for LSTM
def prepare_data(data, look_back=60):
    data = data[['Close']].values
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

# Streamlit UI
st.title("Stock Price Predictor with LSTM")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
period = st.selectbox("Select Data Period:", ["1y", "2y", "5y"], index=0)
look_back = st.slider("Look Back Window (days):", 30, 90, 60)

if st.button("Fetch Data and Train Model"):
    with st.spinner("Fetching data and training model..."):
        df = fetch_data(ticker, period)
        if df.empty:
            st.error("Could not fetch data. Please try a different ticker or check your internet connection.")
        else:
            st.line_chart(df['Close'])
            
            # Prepare data
            X, y = prepare_data(df, look_back)
            
            # Split into train and test (for demo, not production!)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build model (simplified for demo)
            model = Sequential([
                LSTM(30, return_sequences=True, input_shape=(look_back, 1)),
                LSTM(30),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model (for demo, use fewer epochs)
            model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y_test, label='Actual')
            ax.plot(y_pred, label='Predicted')
            ax.legend()
            st.pyplot(fig)
            
            st.success("Model trained and predictions plotted!")
