import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Choose a stock
stock = input("Enter Stock Symbol (e.g., AAPL, TSLA, RELIANCE.NS): ").upper()

# Download last 5 years of data
df = yf.download(stock, period='5y')
if df.empty:
    print("Invalid stock symbol or no data available.")
    exit()

data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

X, y = [], []
for i in range(60, len(scaled_data) - 30):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i:i+30, 0])  # Predict next 30

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(30)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=64, verbose=1)

# Predict next 30 days
last_60_days = scaled_data[-60:]
input_seq = last_60_days.reshape((1, 60, 1))
predicted_scaled = model.predict(input_seq)
predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

print(f"\nPredicted next 30-day prices for {stock}:")
for i, price in enumerate(predicted):
    print(f"Day {i+1}: ₹{price[0]:.2f}")

# Plot results
plt.figure(figsize=(10, 4))
plt.plot(range(1, 31), predicted, marker='o', linestyle='--')
plt.title(f"Predicted 30-Day Prices for {stock}")
plt.xlabel("Future Day")
plt.ylabel("Predicted Price")
plt.grid(True)
plt.tight_layout()
plt.show()

