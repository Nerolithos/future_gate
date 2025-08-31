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


def stream_chat(messages: List[Dict[str, str]], model: str, temperature: float = 0.9):
    """
    Generator that yields incremental text chunks from OpenRouter streaming.
    Usage:
        for chunk in stream_chat(msgs, model, temperature):
            ... append chunk ...
    """
    client = get_openrouter_client()
    if client is None:
        raise RuntimeError("未找到 OpenRouter API Key。请在 Streamlit secrets 中设置 openrouter.api_key，或设置环境变量 OPENROUTER_API_KEY。")
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
    except TypeError:
        # 某些 SDK 版本用 stream=True 参数不兼容；尝试 client.chat.completions.stream
        stream = client.chat.completions.stream(
            model=model,
            messages=messages,
            temperature=temperature,
        )

    # 兼容 openai>=1.0 的事件流接口
    for event in stream:
        try:
            # event 有可能是 delta 事件或包含 choices 数组
            if hasattr(event, "choices") and event.choices:
                delta = getattr(event.choices[0], "delta", None) or getattr(event.choices[0], "message", None)
                if delta:
                    content = getattr(delta, "content", None)
                    if content:
                        yield content
            elif hasattr(event, "data"):
                # 某些实现中 event.data 里有 chunk
                data = event.data
                content = None
                if isinstance(data, dict):
                    content = ((data.get("choices") or [{}])[0].get("delta") or {}).get("content")
                if content:
                    yield content
        except Exception:
            # 安静跳过异常片段
            pass
