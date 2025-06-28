# 📈 Stock Price Predictor with LSTM

This is a Streamlit-based web application that predicts the **next 30 days of stock prices** using historical data and a Long Short-Term Memory (LSTM) model. The app uses real-time data fetched from Yahoo Finance and runs an LSTM neural network under the hood.
# 📈 Stock Price Predictor with LSTM

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Deployed-success?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A real-time stock price forecasting web app powered by LSTM (Long Short-Term Memory), deployed using Streamlit. Enter any stock ticker (like AAPL, TSLA, RELIANCE.NS) and get predictions for the next 30 days.

---

## 🚀 Demo

🔗 [Live App on Streamlit](https://stock-lstm-predictor-th9h3mlyvap6v9ylx2gda3.streamlit.app/)

## 🧠 Features

- Input any stock symbol (e.g., AAPL, TSLA, RELIANCE.NS)
- Fetches past 5 years of daily closing prices
- Scales the data using MinMaxScaler
- Trains a two-layer LSTM model on last 60-day windows
- Predicts stock prices for the next 30 days
- Plots a clear graph of predicted prices
- Fully deployed and usable via Streamlit Cloud

---

## 📚 Theoretical Background

### 🔄 Time Series Forecasting-

Time series data is a sequence of data points collected over consistent time intervals such as, daily stock closing prices. 
In stock market prediction, understanding patterns, seasonality, and trends is crucial.

### 🧠 What is LSTM?

**Long Short-Term Memory (LSTM)** is a special type of Recurrent Neural Network (RNN) that can learn long-term dependencies in sequential data which is perfect for financial time series like: stock prices.

**LSTM solves two major problems of standard RNNs:**

1. **Vanishing gradient problem** – where model forgets earlier data
2. **Short memory** – inability to remember long-term patterns

> 📌 In this project, LSTM is used to learn from past 60 days and recursively forecast next 30 days.

---

### ⚙️ LSTM Model Architecture in This Project

- **Input Layer**: 60 previous closing prices
- **LSTM Layer 1**: 50 units, returns sequences
- **LSTM Layer 2**: 50 units
- **Dense Layer**: 1 unit (output prediction)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)

---
### 🔁 Recursive Forecasting

Each day’s prediction feeds into the next input to forecast multiple days (auto-regressive style).

---

## 🧪 Tech Stack

- Python 3.10
- TensorFlow / Keras
- Streamlit
- yFinance
- NumPy & Pandas
- Matplotlib
- scikit-learn

---

This project is licensed under the MIT License.

Built with ❤️ by @aashvixcodes


