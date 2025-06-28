# backtester.py

import pandas as pd

def backtest_buy_signals(df, investment=10000, holding_days=10):
    df = df.copy()
    
    trades = []

    signal_indices = df[df["Buy_Signal"] == True].index

    for signal_idx in signal_indices:
        
        entry_date = df.iloc[signal_idx]["Date"]
        entry_price = float(df.iloc[signal_idx]["Close"])


        # Define exit point (10 days later or end of data)
        exit_idx = signal_idx + holding_days
        if exit_idx >= len(df):
            exit_idx = len(df) - 1

        

        exit_date = df.iloc[exit_idx]["Date"]
        exit_price = float(df.iloc[exit_idx]["Close"])

        # Calculate return
        shares = investment / entry_price
        pnl = (exit_price - entry_price) * shares
        return_pct = ((exit_price - entry_price) / entry_price) * 100

        trades.append({
            "Entry_Date": entry_date,
            "Exit_Date": exit_date,
            "Entry_Price": round(entry_price, 2),
            "Exit_Price": round(exit_price, 2),
            "PnL (₹)": round(pnl, 2),
            "Return (%)": round(return_pct, 2)
        })

    return pd.DataFrame(trades)



