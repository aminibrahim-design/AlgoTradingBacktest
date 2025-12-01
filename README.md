# AlgoTradingBacktest

Interactive Streamlit app for backtesting simple algorithmic trading strategies on historical price data.

The app lets you:

- Pull OHLCV data from Yahoo Finance
- Configure parameters for several technical strategies
- Run backtests vs. a buy-and-hold benchmark
- Inspect equity curves, trades, and summary performance metrics

> ⚠️ **Disclaimer**  
> This project is for educational and research purposes only.  
> It is **not** financial advice and should not be used to make live trading decisions.

---

## Features

- 📈 **Download market data** with [yfinance] from Yahoo Finance
- 🧮 **Technical indicators** with [pandas-ta]
- ⚙️ **Configurable strategy parameters** via Streamlit sidebar
- 🔁 **Vectorized backtesting engine** for long / flat strategies
- 📊 **Performance analytics**, for example:
  - Total return
  - Annualized (CAGR) return
  - Volatility
  - Sharpe ratio
  - Max drawdown
  - Win rate
- 🔍 **Trade list**: entry / exit dates, returns, and holding periods

---

## Implemented Strategies (Overview)

The app is structured to support multiple rule-based strategies, including:

- **EMA Crossover**
  - Go long when a fast EMA crosses above a slow EMA
  - Exit when the fast EMA crosses back below the slow EMA

- **Bollinger Band Strategy**
  - Uses moving average ± N standard deviations
  - Typical rules: buy near lower band, sell/exit near middle or upper band

- **Z-Score Mean Reversion**
  - Normalize deviations from a moving average using Z-scores
  - Enter when |Z| is above a threshold and revert back toward the mean

- **Pairs Trading (Spread & Z-Score)**
  - Build a spread between two correlated tickers (e.g., SPY vs. IVV)
  - Compute Z-score of the spread over a lookback window
  - Enter long/short legs when Z exceeds entry threshold and exit on reversion

> The exact strategies and parameters are defined in `app.py` and can be customized there.

---

## Tech Stack

- **Python**
- **Streamlit** – UI framework
- **yfinance** – data download
- **pandas / NumPy** – data wrangling
- **pandas-ta** – technical indicators
- **matplotlib / Plotly** (if used in `app.py`) – charts

All Python dependencies are listed in [`requirements.txt`](requirements.txt).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ROCCYK/AlgoTradingBacktest.git
cd AlgoTradingBacktest
