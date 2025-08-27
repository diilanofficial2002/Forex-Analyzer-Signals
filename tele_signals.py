# tele_signals.py
# Telegram Notifier + Typhoon API thin client (optional secondary model).

import os
import requests
from datetime import date

def log_today(message: str, filename: str = "data/daily_log.txt"):
    """
    Append message into a daily file; overwrite if the date header has changed.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        file.write('') 
        
    today_str = date.today().isoformat()
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(today_str + "\n")
            f.write(message + "\n")
        return

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines or not lines[0].strip().startswith(today_str):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(today_str + "\n")
            f.write(message + "\n")
    else:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(message + "\n")


class TelegramNotifier:
    """Simple Telegram Bot API wrapper."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_message(self, text: str):
        api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
        r = requests.post(api_url, json=data, timeout=20)
        r.raise_for_status()
        return r.json()


class TyphoonForexAnalyzer:
    """
    Typhoon API v1 client. Provide a system/user prompt and return model content (str).
    """

    def __init__(self, api_key: str, model: str = "typhoon-v2.1-12b-instruct",
                 base_url: str = "https://api.opentyphoon.ai/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/chat/completions"

    def analyze(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
        }
        resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content
