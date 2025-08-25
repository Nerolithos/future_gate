"""
Shared AI utilities for OpenRouter access.

Secrets precedence (for Streamlit Cloud deployment):
- st.secrets["openrouter"]["api_key"] (recommended)
- st.secrets["OPENROUTER_API_KEY"] (legacy)
- environment variable OPENROUTER_API_KEY (fallback)
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
    # Try nested secrets first: st.secrets["openrouter"]["api_key"]
    try:
        if isinstance(st.secrets, dict):
            key = st.secrets.get("openrouter", {}).get("api_key", "") or st.secrets.get("OPENROUTER_API_KEY", "")
        else:
            # st.secrets is a Secrets object; support attribute/index access
            key = (
                (st.secrets["openrouter"]["api_key"] if "openrouter" in st.secrets else "")
                or st.secrets.get("OPENROUTER_API_KEY", "")
            )
    except Exception:
        key = ""
    if not key:
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
        raise RuntimeError("未找到 OpenRouter API Key。请在 Streamlit secrets 中设置 openrouter.api_key，或设置环境变量 OPENROUTER_API_KEY。")
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return (resp.choices[0].message.content or "").strip()
