import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# Set Streamlit page config
st.set_page_config(page_title="Stock Forecast", layout="centered")
st.title("📈 30-Day Stock Price Predictor using LSTM (PyTorch)")

# Input
stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, RELIANCE.NS)", "AAPL")

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

if st.button("🔮 Predict"):
    with st.spinner("Fetching data and training model..."):
        df = yf.download(stock)
        if df.empty:
            st.error("Invalid symbol or no data found!")
        else:
            df = df[['Close']]
            data = df.values

            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            # Prepare training data
            X, y = [], []
            for i in range(60, len(scaled_data)):
                X.append(scaled_data[i-60:i])
                y.append(scaled_data[i])
            X = np.array(X)
            y = np.array(y)

            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)

            # Define model
            model = LSTMModel()
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Train model
            model.train()
            for epoch in range(5):
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Predict next 30 days
            model.eval()
            last_seq = scaled_data[-60:]
            predictions = []
            for _ in range(30):
                seq_input = torch.FloatTensor(last_seq.reshape(1, 60, 1))
                with torch.no_grad():
                    pred = model(seq_input).item()
                predictions.append(pred)
                last_seq = np.append(last_seq[1:], [[pred]], axis=0)

            predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

            # Output results
            st.success("✅ Prediction Complete!")
            st.subheader(f"Predicted Prices for {stock.upper()}")

            for i, price in enumerate(predicted_prices):
                st.write(f"Day {i+1}: ₹{price[0]:.2f}")

            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, 31), predicted_prices, marker='o', linestyle='--', color='green')
            ax.set_title(f"Predicted Next 30 Days for {stock.upper()}")
            ax.set_xlabel("Day")
            ax.set_ylabel("Price (₹)")
            ax.grid(True)
            st.pyplot(fig)
