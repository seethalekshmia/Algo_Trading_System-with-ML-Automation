# Algorithmic Trading Engine with Backtesting and Machine Learning Integration

This project is a Python-based algorithmic trading engine that analyzes stock market data, generates trading signals based on technical indicators, backtests strategies, and sends real-time alerts. It also includes a machine learning component to train models for predicting price movements. The results, including trade logs and performance summaries, are automatically logged to a Google Sheet.

## 🚀 Features

  - **Multi-Ticker Analysis**: Fetches historical stock data for multiple tickers using the `yfinance` library.
  - **Technical Indicators**: Automatically calculates key indicators, including:
      - Relative Strength Index (RSI)
      - Simple Moving Averages (SMA - 20 and 50 periods)
      - Moving Average Convergence Divergence (MACD)
  - **Strategy & Signal Generation**: Implements a trading strategy based on the MACD crossover to generate buy signals.
  - **Backtesting Engine**: Simulates trades based on generated signals to evaluate the strategy's historical performance, calculating total Profit & Loss, average return, and win ratio.
  - **Machine Learning Models**: Trains and evaluates two types of predictive models:
      - Decision Tree Classifier
      - Logistic Regression
  - **Google Sheets Integration**: Automatically logs detailed trade-by-trade results and a performance summary for each stock to a specified Google Sheet.
  - **Real-time Alerts**: Sends immediate buy signal notifications to a Telegram channel.

## 📁 Project Structure

```
algo_trading_project/
├── main.py                  # Main entry point to run the application
├── algo_engine.py           # Core orchestration logic
├── config.py                # Configuration (tickers, API keys, etc.)
├── data_fetcher.py          # Fetches historical stock data
├── indicators.py            # Calculates technical indicators
├── strategy.py              # Implements the trading strategy and generates signals
├── backtester.py            # Performs backtesting of the strategy
├── ml_model.py              # Handles training and evaluation of ML models
├── sheet_logger.py          # Manages Google Sheets integration
├── telegram_alert.py        # Manages Telegram notifications
├── requirements.txt         # List of Python dependencies
└── README.md                # Project documentation
```

## ⚙️ How It Works

The project executes a sequence of tasks orchestrated by `algo_engine.py`:

1.  **Data Fetching**: For each ticker defined in `config.py`, historical data is downloaded from Yahoo Finance.
2.  **Indicator Calculation**: Technical indicators (RSI, SMA, MACD) are computed and added to the dataset.
3.  **Signal Generation**: The `strategy.py` module analyzes the indicators. A **buy signal** is generated when the MACD line crosses above the MACD signal line.
4.  **Telegram Alerts**: Upon generating a buy signal, a real-time notification is sent via Telegram with the stock ticker, date, and price.
5.  **Backtesting**: The `backtester.py` evaluates the generated signals against historical data. For each signal, it simulates buying the stock and holding it for a predefined number of days (`holding_days`), then calculates the resulting profit or loss.
6.  **Reporting**: The results of the backtest, including a detailed trade log and a high-level performance summary (Total P\&L, Average Return, Win Ratio), are uploaded to separate tabs in a Google Sheet for each ticker.
7.  **Machine Learning (Optional)**: The `ml_model.py` script can be run independently to train models on historical data. It uses features like RSI, MACD, and SMAs to predict whether the next day's closing price will be higher or lower. The trained models are saved for future use.

## 🛠️ Setup and Usage

### 1\. Prerequisites

  - Python 3.x
  - A Google Cloud Platform (GCP) project with the **Google Sheets API** and **Google Drive API** enabled.
  - A Telegram Bot and its API Token.

### 2\. Installation

Clone the repository and install the required dependencies:

```bash
git clone <https://github.com/seethalekshmia/Algo_Trading_System-with-ML-Automation.git>
cd algo_trading_project
pip install -r requirements.txt
```

### 3\. Configuration

1.  **Google Credentials**:

      - Follow the instructions to create a service account and download its JSON credentials file.
      - Rename the file to `credentials.json` and place it in the project's root directory.
      - Share your Google Sheet with the `client_email` found in the `credentials.json` file.

2.  **Environment Variables**:

      - Create a `.env` file in the root directory.
      - Add your Telegram Bot Token and Chat ID to the `.env` file:
        ```
        YOUR_BOT_TOKEN="<your_telegram_bot_token>"
        YOUR_CHAT_ID="<your_telegram_chat_id>"
        ```

3.  **`config.py`**:

      - Open `config.py` and modify the following variables according to your preferences:
          - `TICKERS`: A list of stock tickers to analyze (e.g., `["RELIANCE.NS", "TCS.NS"]`).
          - `DATA_PERIOD_BACKTEST`: The time period for backtesting data (e.g., `"6mo"`).
          - `GOOGLE_SHEET_NAME`: The name of the Google Sheet you want to log results to.

### 4\. Running the Application

  - **To run the main trading algorithm (backtesting and reporting)**:

    ```bash
    python main.py
    ```

    *(This will execute the `run_algo()` function in `algo_engine.py`)*

  - **To train the machine learning models**:

    ```bash
    python ml_model.py
    ```

    *(This will train the Decision Tree and Logistic Regression models and save them as `.pkl` files.)*

## 📜 Dependencies

  - `pandas`
  - `yfinance`
  - `gspread`
  - `oauth2client`
  - `scikit-learn`
  - `joblib`
  - `requests`
  - `python-dotenv`

All dependencies are listed in the `requirements.txt` file.