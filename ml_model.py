# # ml_model.py

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from data_fetcher import fetch_stock_data
# from indicators import apply_indicators
# from strategy import generate_buy_signals
# from config import TICKERS

# def train_ml_model():
#     dfs = []
#     for ticker in TICKERS:
#         df = fetch_stock_data(ticker)
#         df = apply_indicators(df)
#         df['Ticker'] = ticker
#         # close_col = f'Close_{ticker}'
#         # print("Ticker:", ticker)
#         # print("\n🤖", df.columns)
#         # df['Target'] = (df[close_col].shift(-1) > df[close_col]).astype(int)
#         dfs.append(df)
#     # print("\n🤖", dfs.columns)

#     all_df = pd.concat(dfs, ignore_index=True)
# # Flatten MultiIndex columns if present
#     if isinstance(all_df.columns, pd.MultiIndex):
#         all_df.columns = ['_'.join([str(i) for i in col if i]) for col in all_df.columns]

#     # print("\n🤖", all_df.columns)

#     # # Create Target for each ticker
#     # all_df['Target'] = all_df.groupby('Ticker')['Close'].shift(-1) > all_df['Close']
#     # all_df['Target'] = all_df['Target'].astype(int)
#     # all_df = all_df.dropna(subset=['RSI', 'MACD', 'MACD_Signal', 'Volume', 'SMA_20', 'SMA_50', 'Target'])

    

#     # # Target: 1 if next day’s close > today’s, else 0
#     # # all_df["Target"] = (all_df["Close"].shift(-1) > all_df["Close"]).astype(int)
#     # # Drop last row (target would be NaN)
#     # # df = df[:-1]

#     # # Feature set
#     # # features = ["RSI", "MACD", "MACD_Signal", "Volume", "SMA_20", "SMA_50"]
#     # features = [
#     #     'RSI',
#     #     'MACD',
#     #     'MACD_Signal',
#     #     'Volume',
#     #     'SMA_20',
#     #     'SMA_50'
#     # ] +[col for col in all_df.columns if col.startswith('Ticker_')]
#     # # df = df.dropna(subset=features + ["Target"])

#     # X = df[features]
#     # y = df["Target"]


#  # Melt the DataFrame so each row is for one ticker at one time
#     # We'll use only the columns that are common for all tickers
#     melted = pd.DataFrame()
#     for ticker in TICKERS:
#         sub = all_df[[
#             'Date',
#             f'Close_{ticker}',
#             f'Volume_{ticker}',
#             'RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50'
#         ]].copy()
#         sub = sub.rename(columns={
#             f'Close_{ticker}': 'Close',
#             f'Volume_{ticker}': 'Volume'
#         })
#         sub['Ticker'] = ticker
#         melted = pd.concat([melted, sub], ignore_index=True)

#     # Create Target for each ticker
#     melted['Target'] = melted.groupby('Ticker')['Close'].shift(-1) > melted['Close']
#     melted['Target'] = melted['Target'].astype(int)
#     melted = melted.dropna(subset=['RSI', 'MACD', 'MACD_Signal', 'Volume', 'SMA_20', 'SMA_50', 'Target'])

#     # One-hot encode Ticker
#     # melted = pd.get_dummies(melted, columns=['Ticker'])

#     features = [
#         'RSI', 'MACD', 'MACD_Signal', 'Volume', 'SMA_20', 'SMA_50'
#     ] + [col for col in melted.columns if col.startswith('Ticker_')]

#     X = melted[features]
#     y = melted['Target']









#     # Train-test split (80–20)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#     # Train model
#     model = DecisionTreeClassifier(random_state=42)
#     model.fit(X_train, y_train)

#     # Predict and evaluate
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)

#     print("\n🎯 ML Model Accuracy:", round(acc * 100, 2), "%")
#     print("\n📊 Classification Report:")
#     print(classification_report(y_test, y_pred))

#     return model


# model = train_ml_model()










# train_ml_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

from data_fetcher import fetch_stock_data
from indicators import apply_indicators
from config import TICKERS, DATA_PERIOD_MODEL_TRAINING

def train_ml_models(model_dir="models"):
    all_data = []
    data_period = DATA_PERIOD_MODEL_TRAINING
    for ticker in TICKERS:
        df = fetch_stock_data(ticker, period=data_period)
        df = apply_indicators(df)
        df["Ticker"] = ticker

        # Target: 1 if next day's close > today’s
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df[:-1]  # Drop last row (NaN target)

        all_data.append(df)

    # Combine all stocks into one long-form dataset
    df_all = pd.concat(all_data, ignore_index=True)

    # Flatten MultiIndex if present
    # Flatten MultiIndex column names (if needed)
    if isinstance(df_all.columns, pd.MultiIndex):
        df_all.columns = ['_'.join([str(c) for c in col if c]) for col in df_all.columns]


    print("\n🤖 Combined DataFrame Columns:", df_all.columns)

    # Drop rows with missing indicator data
    feature_cols = ["RSI", "MACD", "MACD_Signal", f"Volume_{ticker}", "SMA_20", "SMA_50"]
    df_all = df_all.dropna(subset=feature_cols + ["Target"])

    # One-hot encode ticker
    df_all = pd.get_dummies(df_all, columns=["Ticker"])

    # Feature matrix and target
    X = df_all[feature_cols + [col for col in df_all.columns if col.startswith("Ticker_")]]
    y = df_all["Target"]

    # Train-test split (time series split preferred)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    results = {}

    ### MODEL 1: Decision Tree ###
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    results['Decision Tree'] = {
        "model": dt_model,
        "accuracy": accuracy_score(y_test, y_pred_dt),
        "report": classification_report(y_test, y_pred_dt, output_dict=True)
    }

    # Save model
    joblib.dump(dt_model, f"{model_dir}/decision_tree_model.pkl")


    ### MODEL 2: Logistic Regression ###
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Required for LR
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)

    results['Logistic Regression'] = {
        "model": lr_pipeline,
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "report": classification_report(y_test, y_pred_lr, output_dict=True)
    }

    # Save model
    joblib.dump(lr_pipeline, f"{model_dir}/logistic_regression_model.pkl")

    # Print results
    for model_name, data in results.items():
        print(f"\n🔍 {model_name}")
        print(f"🎯 Accuracy: {round(data['accuracy'] * 100, 2)}%")
        print("📊 Classification Report:")
        print(classification_report(y_test, data['model'].predict(X_test)))

    return results
if __name__ == "__main__":
    train_ml_models()
    print("\n✅ ML Models trained and saved successfully!")