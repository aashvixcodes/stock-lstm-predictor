# 📈 Stock Price Predictor with LSTM

This is a Streamlit-based web application that predicts the **next 30 days of stock prices** using historical data and a Long Short-Term Memory (LSTM) model. The app uses real-time data fetched from Yahoo Finance and runs an LSTM neural network under the hood.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Local--Run-informational?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🚀 Run This App Locally

### 🔧 Prerequisites

- Python 3.10
- pip installed

### 🛠 Steps

```bash
# 1. Clone the repository
git clone https://github.com/aashvixcodes/stock-lstm-predictor.git
cd stock-lstm-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py


## 🧠 Features

- Input any stock symbol (e.g., AAPL, TSLA, RELIANCE.NS)
- Fetches past 5 years of daily closing prices
- Scales the data using MinMaxScaler
- Trains a two-layer LSTM model on last 60-day windows
- Predicts stock prices for the next 30 days
- Plots a clear graph of predicted prices

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

## 🛠 Tech Stack Used

Here's a breakdown of the technologies and tools used to build and deploy this project:

### 📊 Data Handling & Processing
- **NumPy** – for array manipulation and numerical computations
- **Pandas** – for working with stock price data in tabular format
- **yfinance** – to fetch historical stock data from Yahoo Finance

### 🧠 Machine Learning Model

- **Model Type**: Long Short-Term Memory (LSTM)  
- **Framework**: TensorFlow/Keras  

### 📈 Visualization
- **Matplotlib** – to visualize the predicted stock prices

### 🌐 Web App Interface
- **Streamlit** – to create and host the interactive web app

### ⚙️ Deployment & DevOps
- **Git & GitHub** – for version control and cloud hosting the repository
- **Streamlit Cloud** – to deploy the Streamlit app live from GitHub

### 💻 Environment
- **Python 3.10** – compatible with TensorFlow and Streamlit Cloud

---

### ✅ Summary Table

| Layer               | Technology / Library                  |
|--------------------|----------------------------------------|
| Data Fetching      | `yfinance`                             |
| Data Processing    | `pandas`, `numpy`, `MinMaxScaler`      |
| ML Model           | `sklearn  ` `tensorflow.keras`         |                   
| Visualization      | `matplotlib`, `streamlit.pyplot`       |
| Frontend Interface | `Streamlit`                            |
| Deployment         | `GitHub`, `Streamlit Cloud`            |
| Python Version     | `Python 3.10`                          |

---

This project is licensed under the MIT License.

Built with ❤️ by @aashvixcodes


