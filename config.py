# config.py

TICKERS = ["TCS.NS", "TITAN.NS", "RELIANCE.NS"]
DATA_PERIOD_BACKTEST = "6mo"  # Options: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
DATA_PERIOD_MODEL_TRAINING = "10y" 
DATA_INTERVAL = "1d"  # Options: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "1wk", "1mo"

GOOGLE_SHEET_NAME = "Algo_Trading_Log"
CREDENTIALS_FILE = "google_credentials.json"  # Download from Google Cloud Console


