# -*- coding: utf-8 -*-
"""
CUHKSZ Q&A feature. Exposes page_cuhksz_mode().
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict

import streamlit as st

from ai_utils import generate_chat


BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "Prompt"


def read_prompt_templates() -> Dict[str, str]:
    templates: Dict[str, str] = {}
    if PROMPT_DIR.exists():
        for p in list(PROMPT_DIR.glob("*.md")) + list(PROMPT_DIR.glob("*.txt")):
            try:
                templates[p.stem] = p.read_text(encoding="utf-8")
            except Exception:
                pass
    return templates


def build_cuhksz_prompt(primary: str, secondary: str, template_text: str, user_extra: str) -> str:
    header = ("你是一位熟悉 CUHKSZ（香港中文大学（深圳））校园与学术生态的助理，回答用中文，具体且可操作。"
              "先列要点清单，再给出完整说明，必要时用步骤/表格。")
    route = f"问题类别路径：{primary} → {secondary}\n"
    extra = f"\n用户补充信息：\n{user_extra.strip()}\n" if user_extra.strip() else ""
    return f"{header}\n{route}\n模板：\n{template_text.strip()}\n{extra}"


def page_cuhksz_mode():
    st.markdown("### 你想咨询？")
    primary = st.radio("第一层", ["生活", "技术"], horizontal=True, key="cuhksz_primary")
    if primary == "生活":
        secondary = st.radio("第二层（生活）", ["选课相关", "日常相关"], horizontal=True, key="cuhksz_secondary_life")
    else:
        secondary = st.radio("第二层（技术）", ["CSC 系列课程相关", "课外内容"], horizontal=True, key="cuhksz_secondary_tech")

    templates = read_prompt_templates()
    st.markdown("---")
    st.markdown("**模板来源：** 自动从 `Prompt/` 目录读取 `.md`/`.txt`。缺失也可仅用你的补充文本调用模型。")

    suggested_map = {
        ("生活", "选课相关"): ["life_course", "course", "选课"],
        ("生活", "日常相关"): ["life_daily", "daily", "生活"],
        ("技术", "CSC 系列课程相关"): ["tech_csc", "csc"],
        ("技术", "课外内容"): ["tech_extra", "extra", "课外"],
    }
    suggested = suggested_map.get((primary, secondary), [])
    keys = list(templates.keys()); default = keys[0] if keys else ""
    for k in keys:
        if any(tag in k.lower() for tag in [s.lower() for s in suggested]):
            default = k; break
    tpl_key = st.selectbox("选择模板（来自 Prompt/）", ["（不使用模板）"] + keys,
                           index=0 if not default else (keys.index(default) + 1))
    tpl_text = "" if tpl_key == "（不使用模板）" else templates.get(tpl_key, "")
    user_extra = st.text_area("补充你的问题或上下文（选填）：", height=140,
                              placeholder="例如：我大二，GPA 3.6，想选 XXX 老师；或我想做 CSC300X 的项目，方向是 …")
    final_prompt = build_cuhksz_prompt(primary, secondary, tpl_text, user_extra)
    with st.expander("👉 将发送给模型的 Prompt（可复制）", expanded=False):
        st.code(final_prompt, language="markdown")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("模型（OpenRouter）", ["openai/gpt-5-chat", "openai/gpt-5"], index=0, key="cuhksz_model")
    with col2:
        temp = st.slider("随机性", 0.0, 1.2, 0.7, 0.05, key="cuhksz_temp")

    if st.button("🚀 生成回答", use_container_width=True):
        try:
            messages = [
                {"role": "system", "content": "You are an expert counselor for CUHKSZ students. Be precise and kind. Answer in Chinese."},
                {"role": "user", "content": final_prompt},
            ]
            st.session_state["cuhksz_answer"] = generate_chat(messages, model=model, temperature=temp)
        except Exception as e:
            st.error(f"生成失败：{e}")

    if ans := st.session_state.get("cuhksz_answer"):
        st.markdown("### 回答"); st.write(ans)
        st.download_button("⬇️ 下载回答", data=ans, file_name="cuhksz_answer.md", mime="text/markdown")
