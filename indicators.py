# indicators.py

import pandas as pd

def calculate_rsi(df, period=14, column="Close"):
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df['RSI'] = rsi
    return df


def calculate_sma(df, period=20, column="Close", label=None):
    if not label:
        label = f"SMA_{period}"
    df[label] = df[column].rolling(window=period).mean()
    return df


def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """
    Calculate MACD and Signal line.
    MACD = EMA12 - EMA26
    Signal line = 9-day EMA of MACD
    """
    df = df.copy()
    df['EMA_12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df


def apply_indicators(df):
    df = calculate_rsi(df)
    df = calculate_sma(df, period=20)
    df = calculate_sma(df, period=50)
    df = calculate_macd(df)     

    return df


# indicators.py

