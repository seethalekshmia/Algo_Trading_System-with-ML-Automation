# algo_engine.py

from data_fetcher import fetch_stock_data
from indicators import apply_indicators
from strategy import generate_buy_signals
from backtester import backtest_buy_signals
from sheet_logger import connect_to_sheet, update_sheet_with_df
from config import TICKERS, DATA_PERIOD_BACKTEST 

import pandas as pd


def run_algo(sheet_name="Algo Trading Report", investment=10000, holding_days=10):
    print("\n🚀 Starting Algo Scan...")

    sheet = connect_to_sheet(sheet_name)
    data_period_backtest= DATA_PERIOD_BACKTEST
    for ticker in TICKERS:
        print(f"\n🔎 Processing {ticker}...")
        # 1. Fetch historical data

        df = fetch_stock_data(ticker, period=data_period_backtest)
        df = apply_indicators(df)
        df = generate_buy_signals(df)
        
        bt_result = backtest_buy_signals(df, investment, holding_days)

        if not bt_result.empty:

            summary_df = pd.DataFrame([{
                "Stock": ticker,
                "Total P&L (₹)": bt_result["PnL (₹)"].sum(),
                "Average Return (%)": bt_result["Return (%)"].mean(),
                "Win Ratio": (bt_result["PnL (₹)"] > 0).mean()
            }])
            summary_df = summary_df.astype(str)
            bt_result = bt_result.astype(str)


            update_sheet_with_df(sheet, f"{ticker}_Trade_Log", bt_result)

            update_sheet_with_df(sheet, f"{ticker}_Summary", summary_df)
        else:
            print(f"⚠️ No signals found for {ticker}. Skipping sheet update.")

    print("\n✅ Algo Scan Complete!")
