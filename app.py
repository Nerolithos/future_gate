# -*- coding: utf-8 -*-
"""
AI 辅助龙大生存指南与故事编写
- 登录：从 Streamlit 的 st.secrets（auth.users 或根级 users）读取用户表
- 密码正确后进行“表情验证码”（摄像头 + OpenRouter 视觉模型）
- 登录后进入首页：① AI 写故事（0~5 层 + 结局 + 小说生成）② 提问 CUHKSZ（两层选择 + Prompt 模板）

模块说明：
- auth.py：登录与 AI 表情验证码
- futuregate.py：AI 生成故事功能
- cuhksz.py：提问 CUHKSZ 功能
"""

from __future__ import annotations
import streamlit as st

from auth import login_gate
from futuregate import page_story_mode
from cuhksz import page_cuhksz_mode

# ==== 页面设置 ====
st.set_page_config(page_title="AI 辅助龙大生存指南与故事编写", page_icon="🤖", layout="wide")

# ==== 首页 ====
def main():
    # 登录拦截（含表情验证码）
    if not login_gate():
        return

    # 顶部标题（登录后）
    st.title("AI 辅助龙大生存指南与故事编写")
    user_display = st.session_state.get('username','') or ("访客" if st.session_state.get('guest_login') else "")
    st.caption(f"欢迎，{user_display} · OpenRouter 驱动")

    # 退出登录（侧边）
    with st.sidebar:
        if st.button("退出登录"):
            st.session_state.clear(); st.rerun()

    # 模式选择
    try:
        mode = st.segmented_control("选择模式：", options=["AI 写故事", "提问 CUHKSZ"], default="AI 写故事")
    except Exception:
        mode = st.radio("选择模式：", ["AI 写故事", "提问 CUHKSZ"], horizontal=True, index=0)

    if mode == "AI 写故事":
        page_story_mode()
    else:
        page_cuhksz_mode()

if __name__ == "__main__":
    main()