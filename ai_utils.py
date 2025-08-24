"""
Shared AI utilities for OpenRouter access.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Any
import os

import streamlit as st

try:
    from openai import OpenAI  # type: ignore
    _SDK_OK = True
except Exception:
    OpenAI = None  # type: ignore
    _SDK_OK = False


def get_openrouter_client() -> Optional[Any]:
    if not _SDK_OK:
        return None
    key = ""
    try:
        key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return None
    headers = {
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501"),
        "X-Title": "FutureGate-Life3",
    }
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, default_headers=headers)


def generate_chat(messages: List[Dict[str, str]], model: str, temperature: float = 0.9) -> str:
    client = get_openrouter_client()
    if client is None:
        raise RuntimeError("未找到 OPENROUTER_API_KEY。请在 .streamlit/secrets.toml 或环境变量中配置。")
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return (resp.choices[0].message.content or "").strip()
