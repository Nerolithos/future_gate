# -*- coding: utf-8 -*-
"""
Authentication and AI captcha (vision expression) flow for Streamlit app.
Exposes: login_gate()
"""
from __future__ import annotations
from typing import Optional, Dict, Tuple
import os, io, base64, json, random

import streamlit as st
from PIL import Image

from ai_utils import get_openrouter_client


LOGIN_TITLE = "AI 辅助龙大生存指南与故事编写"


def _imgfile_to_data_url(uploaded_file) -> str:
    img = Image.open(uploaded_file).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def check_expression_with_openrouter(data_url: str, target_en: str, model: str = "openai/gpt-4o-mini") -> dict:
    client = get_openrouter_client()
    if client is None:
        raise RuntimeError("未配置 OPENROUTER_API_KEY")

    prompt = (
        "You are a strict vision validator for a human-verification task.\n"
        f"Task: Check if the person in the image is performing this expression/pose: {target_en}.\n"
        "Answer in minified JSON only, with keys: match(bool), confidence(0..1), reason(str)."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    }]
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
    text = (resp.choices[0].message.content or "").strip()
    try:
        if text.startswith("{"):
            data = json.loads(text)
        else:
            s, e = text.find("{"), text.rfind("}")
            data = json.loads(text[s:e+1])
    except Exception:
        data = {"match": False, "confidence": 0.0, "reason": "parse_error"}
    data.setdefault("match", False)
    data.setdefault("confidence", 0.0)
    data.setdefault("reason", "")
    return data


EXPRESSIONS = [
    ("微笑露齿", "a broad smile showing teeth", "😁"),
    ("张大嘴巴", "mouth widely open", "😮"),
    ("交叉手指", "a photo of fingers cross", "🤞"),
    ("抬眉挑眉", "raise both eyebrows noticeably", "🤨"),
    ("吐出舌头", "stick out one's tongue", "😛"),
    ("竖大拇指", "a photo of one's thumb up", "👍"),
    ("安静的手势", "make the hand sign of quiet (shhhh)(stick one finger up) infront of one's mouth", "🤫"),
]


def inject_login_bg():
    st.markdown(
        """
        <style>
        [data-testid="stHeader"] { background: transparent !important; }
        [data-testid="stAppViewContainer"]{
          background:
            radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.05) 0, transparent 60%) ,
            radial-gradient(1px 1px at 80% 70%, rgba(255,255,255,0.04) 0, transparent 60%) ,
            repeating-linear-gradient(0deg, rgba(255,255,255,0.04) 0 1px, transparent 1px 60px),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.03) 0 1px, transparent 1px 60px),
            radial-gradient(900px 900px at 85% 15%, rgba(99,102,241,0.15), transparent 60%),
            radial-gradient(800px 800px at 15% 85%, rgba(16,185,129,0.12), transparent 55%),
            linear-gradient(135deg, #0b1020 0%, #0f172a 40%, #111827 100%);
          background-size: 400% 400%;
          animation: bgShift 26s ease-in-out infinite;
          color: #e5e7eb;
        }
        @keyframes bgShift { 0% {background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
        .login-title { font-size: 56px; line-height: 1.1; font-weight: 900; letter-spacing: .4px; margin-bottom: 8px; color: #F3F4F6; text-shadow: 0 6px 24px rgba(0,0,0,.35); }
        .login-subtitle { font-size: 15px; opacity: .9; margin-bottom: 22px; color: #D1D5DB; }
        .block-container { padding-top: 3rem; }
        .stTextInput > div > div > input { background: #e5e7eb; }
        .stTextInput > div { border-radius: 12px; }
        .stButton > button { border-radius: 12px; height: 44px; font-weight: 700; background: linear-gradient(90deg, #6366F1, #22C55E); color: white; border: none; box-shadow: 0 10px 24px rgba(0,0,0,.25); }
        .stButton > button:hover { filter: brightness(1.05); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def login_gate() -> bool:
    if st.session_state.get("logged_in"):
        return True

    inject_login_bg()

    if st.session_state.get("need_face_captcha"):
        st.markdown(f'<div class="login-title">表情验证:请模仿——</div>', unsafe_allow_html=True)
        target = st.session_state.get("face_target")
        if not target:
            target = random.choice(EXPRESSIONS)
            st.session_state["face_target"] = target
        name_cn, target_en, emoji = target
        st.markdown(
            f"""
            <p style="color:#9AE6B4; background:rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.35); padding:.75rem 1rem;border-radius:12px; font-weight:700; font-size: 40px; ">
                </b> {emoji}
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("图片仅用于本地验证，不会保存")

        img_file = st.camera_input("点击下方拍摄", key="face_shot")
        model_for_vision = "openai/gpt-4o-mini"
        conf_bar = 0.75

        if img_file:
            try:
                data_url = _imgfile_to_data_url(img_file)
                res = check_expression_with_openrouter(data_url, target_en, model=model_for_vision)
                passed = bool(res.get("match")) and float(res.get("confidence", 0)) >= conf_bar
                st.write(f"模型判断：`match={res.get('match')}`, 置信度={float(res.get('confidence', 0.0)):.2f}")
                if res.get("reason"):
                    st.caption(res["reason"])
                if passed:
                    st.success("表情验证通过 ✅")
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = st.session_state.get("pending_user", "")
                    for k in ("need_face_captcha", "face_target", "pending_user"):
                        st.session_state.pop(k, None)
                    st.rerun()
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("换一个表情重试"):
                            st.session_state.pop("face_target", None)
                            st.rerun()
                    with c2:
                        if st.button("返回登录"):
                            for k in ("need_face_captcha", "face_target", "pending_user"):
                                st.session_state.pop(k, None)
                            st.rerun()
            except Exception as e:
                st.error(f"验证失败：{e}")

        st.stop()

    colL, colC, colR = st.columns([1, 1.1, 1])
    with colC:
        st.markdown(f'<div class="login-title">{LOGIN_TITLE}</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">大模型辅助解答港中深相关问题/大模型辅助编故事（试运行）。</div>', unsafe_allow_html=True)

        username = st.text_input("用户名", key="login_user", placeholder="例如：Jinghong Li")
        password = st.text_input("密码", type="password", key="login_pass", placeholder="请输入 4 位数字密码")
        c1, c2 = st.columns(2)
        with c1:
            submit = st.button("登录", use_container_width=True)
        with c2:
            guest = st.button("无账号登录", use_container_width=True)

        st.caption("需要申请账号，以使用非免费大模型，请向 124090960 发邮件说明")

        try:
            user_table = dict(st.secrets.get("users", {}))
        except Exception:
            user_table = {}

        if submit:
            if username not in user_table:
                if hasattr(st, "modal"):
                    with st.modal("登录失败"):
                        st.error("此用户不存在"); st.button("关闭")
                else:
                    st.error("此用户不存在")
            elif str(user_table[username]) != str(password):
                if hasattr(st, "modal"):
                    with st.modal("登录失败"):
                        st.error("密码错误"); st.button("关闭")
                else:
                    st.error("密码错误")
            else:
                st.session_state["guest_login"] = False
                st.session_state["pending_user"] = username
                st.session_state["need_face_captcha"] = True
                st.rerun()

        # 无账号登录：进入访客模式，同样需要表情验证
        if 'guest' in locals() and guest:
            st.session_state["guest_login"] = True
            st.session_state["pending_user"] = "访客"
            st.session_state["need_face_captcha"] = True
            st.rerun()

    st.stop()
