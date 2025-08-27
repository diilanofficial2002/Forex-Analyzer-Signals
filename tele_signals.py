# tele_signals.py
# Telegram notifier + EOD daily log (write-once-per-day).

import os, json
from datetime import datetime, timedelta, timezone
import requests

BKK_TZ = timezone(timedelta(hours=7))

class TelegramNotifier:
    """Minimal Telegram Bot API sender."""
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_message(self, text: str):
        if not (self.bot_token and self.chat_id):
            return
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": text}
            requests.post(url, json=payload, timeout=15)
        except Exception:
            pass


def log_today_summary_only(text: str, path: str = "data/daily_log.txt"):
    """
    Overwrite today's daily log once at EOD to avoid noisy duplicates.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    today = datetime.now(BKK_TZ).date().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {today}\n")
        f.write(text.strip() + "\n")


# Optional Typhoon analyzer (fallback using OpenAI for compatibility)
class TyphoonForexAnalyzer:
    """
    Lightweight stub that uses OpenAI chat as a fallback "Typhoon-style" analyzer.
    Only used when a TYPHOON_API_KEY is provided and primary parse fails.
    """
    def __init__(self, api_key: str, model: str = "gpt-5-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=900,
        )
        return resp.choices[0].message.content
