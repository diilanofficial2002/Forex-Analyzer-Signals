# tele_signals.py
# Utilities: Telegram notifier, Typhoon "formatter" (summary only), and EOD log helper.

import os
import json
import requests
from datetime import date


def log_today_summary_only(message: str, filename: str = "data/daily_log.txt"):
    """
    Write an EOD summary file ONCE per day:
    - Overwrites the file with today's date at the first line, then the message.
    - Keeps the file small even when the workflow runs many times a day.
    """
    today_str = date.today().isoformat()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(today_str + "\n")
        f.write(message.strip() + "\n")


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_message(self, message: str, parse_mode: str = "Markdown"):
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        resp = requests.post(
            url,
            data={"chat_id": self.chat_id, "text": message, "parse_mode": parse_mode},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()


class TyphoonForexAnalyzer:
    """
    Typhoon wrapper used ONLY for formatting/summarizing GPT's accepted plan into a
    nice Telegram message. It MUST NOT change numbers or trading decisions.

    Two modes:
    1) Custom endpoint (preferred if you host your own):
       - env: TYPHOON_API_URL + TYPHOON_API_KEY
       - POST JSON: {"system": "...", "user": "..."}

    2) OpenTyphoon official API:
       - env: TYPHOON_API_KEY (+ optional TYPHOON_BASE_URL, TYPHOON_MODEL)
       - default base_url: https://api.opentyphoon.ai/v1
       - endpoint: /chat/completions
    """

    def __init__(self, api_key: str):
        self.api_key = (api_key or "").strip()

        # Mode 1: custom endpoint
        self.custom_url = os.getenv("TYPHOON_API_URL", "").strip()

        # Mode 2: OpenTyphoon official
        self.base_url = os.getenv("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1").strip()
        self.model    = os.getenv("TYPHOON_MODEL", "typhoon-v2.1-12b-instruct").strip()
        self._ct_endpoint = f"{self.base_url}/chat/completions"

    def is_configured(self) -> bool:
        """Return True if either custom endpoint or OpenTyphoon API key is present."""
        if self.custom_url and self.api_key:
            return True
        if self.api_key:
            return True
        return False

    # ---------------- Formatter-only interface ----------------

    @staticmethod
    def _formatting_system_prompt() -> str:
        return (
            "You are a formatting engine for Telegram. "
            "Given a JSON plan produced by an upstream model, render a short, clean summary. "
            "IMPORTANT RULES:\n"
            "- DO NOT change any numbers, symbols, order types, or the OCO combo.\n"
            "- Keep prices exactly as provided (no rounding beyond given precision).\n"
            "- Use terse bullet style. Thai language for prose. Keep code/labels as-is.\n"
            "- If the plan is rejected, output a one-line reason only.\n"
        )

    @staticmethod
    def _build_user_prompt(plan_obj: dict, market_meta: dict | None = None) -> str:
        """
        Provide the exact JSON plan and optional meta so the LLM can format a nice message
        without inventing or changing values.
        """
        meta = market_meta or {}
        return (
            "สรุปเป็นข้อความสั้นสำหรับ Telegram จาก JSON ต่อไปนี้ ห้ามเปลี่ยนตัวเลขหรือชนิดออเดอร์:\n\n"
            f"PLAN_JSON:\n```json\n{json.dumps(plan_obj, ensure_ascii=False, indent=2)}\n```\n\n"
            f"META (optional):\n```json\n{json.dumps(meta, ensure_ascii=False)}\n```\n\n"
            "รูปแบบแนะนำ:\n"
            "PAIR + สถานะ + COMBO + ความมั่นใจ + เหตุผลสั้นๆ + รายการออเดอร์ (type, entry, SL, TP, RR)\n"
        )

    def summarize_plan(self, plan_obj: dict, market_meta: dict | None = None) -> str:
        """
        Summarize-only: produce a neat Telegram message from a deterministic plan JSON.
        """
        if not self.is_configured():
            raise RuntimeError("Typhoon is not configured (missing API key and/or endpoint).")

        system_prompt = self._formatting_system_prompt()
        user_prompt   = self._build_user_prompt(plan_obj, market_meta)

        # Mode 1: custom endpoint
        if self.custom_url:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {"system": system_prompt, "user": user_prompt}
            r = requests.post(self.custom_url, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data.get("text") or data.get("output_text") or json.dumps(data, ensure_ascii=False)

        # Mode 2: OpenTyphoon official
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
            "max_tokens": 600,
            "temperature": 0.1,
        }
        r = requests.post(self._ct_endpoint, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
