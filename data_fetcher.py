# data_fetcher.py

import yfinance as yf
import pandas as pd
from config import TICKERS, DATA_PERIOD_BACKTEST, DATA_INTERVAL

def fetch_stock_data(ticker, period, interval=DATA_INTERVAL):
    try:
        df = yf.download(ticker, period=period, interval=interval)
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df['Ticker'] = ticker
        df.to_csv(f"data/{ticker}_data.csv", index=False)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()
