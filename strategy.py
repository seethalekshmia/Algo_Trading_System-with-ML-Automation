# strategy.py

import pandas as pd
from telegram_alert import send_telegram_message  # Ensure this module exists or implement it
import datetime
def generate_buy_signals(df):
    df = df.copy()

    


    # Bullish MACD crossover
    df['MACD_Crossover'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))

    # Buy when MACD crossover happens
    df['Buy_Signal'] = df['MACD_Crossover']

    # Create a 'Signal_Date' column for easy filtering
    df['Signal_Date'] = df['Date'].where(df['Buy_Signal'])
    print("DEBUG columns before melt:", df.columns.tolist())

    df_melted = df.copy()

    # Flatten MultiIndex columns if present
    if isinstance(df_melted.columns, pd.MultiIndex):
        df_melted.columns = ['_'.join([str(i) for i in col if i]) for col in df_melted.columns]

    # Melt the DataFrame to long format for Close columns
    melted = pd.melt(
        df_melted,
        id_vars=['Date', 'Buy_Signal'],
        value_vars=[col for col in df_melted.columns if col.startswith('Close_')],
        var_name='Ticker',
        value_name='Close'
    )
    melted['Ticker'] = melted['Ticker'].str.replace('Close_', '')


    # Only send notifications for rows where Buy_Signal is True and price is not NaN
    for idx, row in melted[(melted['Buy_Signal']) & (~melted['Close'].isna())].iterrows():
        ticker = row['Ticker']
        date = row['Date']
        price = row['Close']
        if hasattr(date, 'strftime'):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        message = (
            f"📈 *Buy Signal Triggered!*\n"
            f"Ticker: {ticker}\n"
            f"Date: {date_str}\n"
            f"Price: ₹{round(float(price), 2)}"
        )
        send_telegram_message(message)
    

    return df
