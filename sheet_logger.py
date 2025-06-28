# sheet_logger.py

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# Setup
def connect_to_sheet(sheet_name, creds_path="credentials.json"):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)
    return client.open(sheet_name)

# Clear and write a DataFrame to a specific worksheet
def update_sheet_with_df(sheet, tab_name, df):
    try:
        # Clean all values: convert Timestamp, float32, int64 → string/float
        df_clean = df.copy()
        df_clean = df_clean.astype(str)
        try:
            worksheet = sheet.worksheet(tab_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sheet.add_worksheet(title=tab_name, rows="100", cols="20")

        worksheet.clear()
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())
        print(f"✅ Updated Google Sheet tab: {tab_name}")
    except Exception as e:
        print(f"❌ Error updating {tab_name}: {e}")
