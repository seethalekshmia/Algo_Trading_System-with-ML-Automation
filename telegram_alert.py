import os
import requests
from dotenv import load_dotenv
load_dotenv()



TELEGRAM_TOKEN = os.getenv("YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("YOUR_CHAT_ID")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    payload = {

        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print("❌ Telegram Error:", response.text)
    except Exception as e:
        print("❌ Telegram Exception:", e)
