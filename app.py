# -*- coding: utf-8 -*-
"""
AI 辅助龙大生存指南与故事编写
- 启动即登录（从 .streamlit/secrets.toml 的 [users] 读取用户/密码）
- 密码正确后进行“表情验证码”（摄像头 + OpenRouter 视觉模型）
- 登录后进入首页：① AI 写故事（0~5 层 + 结局 + 小说生成）② 提问 CUHKSZ（两层选择 + Prompt 模板）
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
import os, glob, re, io, base64, json, random

import yaml
import streamlit as st
from PIL import Image

# ==== 基础路径 ====
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "story.yaml"
PROMPT_DIR = BASE_DIR / "Prompt"

# ==== 页面设置 ====
st.set_page_config(page_title="AI 辅助龙大生存指南与故事编写", page_icon="🤖", layout="wide")

# ===== 故事背景（100字左右） =====
STORY_BACKGROUND = (
    "十年里，算法接管了生产排程与能源调度，淡水与稀土让旧秩序摇晃。"
    "穷人更忙，富人更孤独，城市像会呼吸的机器。有人把希望交给新技术，有人把灵魂交给旧信仰。"
    "你在一个普通清晨醒来，必须回答一串问题：我们把缰绳交给谁，怎样活下去？"
)

# ==== OpenRouter SDK ====
try:
    from openai import OpenAI  # openai>=1.0
    _SDK_OK = True
except Exception:
    OpenAI = None  # type: ignore
    _SDK_OK = False

def get_openrouter_client() -> Optional[OpenAI]:
    if not _SDK_OK:
        return None
    key = ""
    try:
        key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return None
    headers = {  # 仅 ASCII，避免 'ascii' codec 错误
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

# ==== 视觉判定工具（表情验证码） ====
def _imgfile_to_data_url(uploaded_file) -> str:
    """camera_input 返回的文件转为 data:image/png;base64,...（不落盘）"""
    img = Image.open(uploaded_file).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def check_expression_with_openrouter(data_url: str, target_en: str, model: str = "openai/gpt-4o-mini") -> dict:
    """
    让视觉模型判断是否与目标表情/动作匹配。
    返回 dict: {"match": bool, "confidence": float, "reason": "..."}
    """
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
    # 粗鲁但实用的 JSON 提取
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

# 随机目标表情/动作池（中文名, 英文判定指令, emoji）
EXPRESSIONS = [
    ("微笑露齿", "a broad smile showing teeth", "😁"),
    ("张大嘴巴", "mouth widely open", "😮"),
    ("交叉手指", "a photo of fingers cross", "🤞"),
    ("抬眉挑眉", "raise both eyebrows noticeably", "🤨"),
    ("吐出舌头", "stick out one's tongue", "😛"),
    ("竖大拇指", "a photo of one's thumb up", "👍"),
    ("安静的手势", "make the hand sign of quiet (shhhh)(stick one finger up) infront of one's mouth", "🤫")
]

# ==== 公共小函数 ====
@st.cache_data(show_spinner=False)
def load_story() -> Dict:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"未找到 {DATA_FILE}")
    data = yaml.safe_load(DATA_FILE.read_text(encoding="utf-8"))
    if "layers" not in data or "endings" not in data:
        raise ValueError("story.yaml 缺少 layers / endings")
    return data

def path_code(picks: List[bool]) -> str:
    return "".join("T" if x else "F" for x in picks)

def _get_by_prefix(mapping: Dict[str, Dict], prefix: str) -> Dict:
    if prefix in mapping:
        return mapping[prefix]
    for n in range(len(prefix) - 1, 0, -1):
        sub = prefix[:n]
        if sub in mapping:
            return mapping[sub]
    return mapping[prefix]  # 故意触发 KeyError 暴露 YAML 错误

def get_layer(layers: List[Dict], depth: int, prefix: str) -> Tuple[str, str, str]:
    layer = layers[depth]
    if "question" in layer:
        q = layer["question"]; opts = layer["options"]
        return q, opts["T"], opts["F"]
    q_map = layer.get("question_by_path", {}); o_map = layer.get("options_by_path", {})
    q_entry = _get_by_prefix(q_map, prefix); o_entry = _get_by_prefix(o_map, prefix)
    return q_entry, o_entry["T"], o_entry["F"]

def decisions_summary(layers: List[Dict], picks: List[bool]) -> List[Tuple[str, str]]:
    summary: List[Tuple[str, str]] = []; prefix = ""
    for depth, ans in enumerate(picks):
        q, lt, lf = get_layer(layers, depth, prefix if prefix else "")
        summary.append((q, lt if ans else lf)); prefix += "T" if ans else "F"
    return summary

def get_snapshot(snapshots: Dict[str, str], prefix: str) -> Optional[str]:
    return snapshots.get(prefix)

def beijing_date_str() -> str:
    dt = datetime.now(ZoneInfo("Asia/Shanghai"))
    return f"{dt.year}年{dt.month}月{dt.day}日"

def read_prompt_templates() -> Dict[str, str]:
    templates: Dict[str, str] = {}
    if PROMPT_DIR.exists():
        for p in list(PROMPT_DIR.glob("*.md")) + list(PROMPT_DIR.glob("*.txt")):
            try: templates[p.stem] = p.read_text(encoding="utf-8")
            except Exception: pass
    return templates

# ==== 登录页样式（背景挂到 ViewContainer；移除“空框”） ====
LOGIN_TITLE = "AI 辅助龙大生存指南与故事编写"

def inject_login_bg():
    st.markdown("""
    <style>
    /* 透明化头部，避免挡住背景 */
    [data-testid="stHeader"] { background: transparent !important; }

    /* 真正的页面容器：挂背景在这里，确保可见 */
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

    /* 登录区域的标题更醒目 */
    .login-title {
      font-size: 56px;
      line-height: 1.1;
      font-weight: 900;
      letter-spacing: .4px;
      margin-bottom: 8px;
      color: #F3F4F6;
      text-shadow: 0 6px 24px rgba(0,0,0,.35);
    }
    .login-subtitle {
      font-size: 15px;
      opacity: .9;
      margin-bottom: 22px;
      color: #D1D5DB;
    }

    .block-container { padding-top: 3rem; }

    .stTextInput > div > div > input { background: #e5e7eb; }
    .stTextInput > div { border-radius: 12px; }
    .stButton > button {
      border-radius: 12px; height: 44px; font-weight: 700;
      background: linear-gradient(90deg, #6366F1, #22C55E);
      color: white; border: none;
      box-shadow: 0 10px 24px rgba(0,0,0,.25);
    }
    .stButton > button:hover { filter: brightness(1.05); }
    </style>
    """, unsafe_allow_html=True)

# ==== 登录关卡（加入“表情验证码”二次步骤） ====
def login_gate() -> bool:
    """未登录则显示登录页/表情验证；登录后返回 True 并继续渲染主界面。"""
    # 已登录
    if st.session_state.get("logged_in"):
        return True

    inject_login_bg()

    # 若处于表情验证阶段，直接渲染表情验证码
    if st.session_state.get("need_face_captcha"):
        st.markdown(f'<div class="login-title">表情验证:请模仿——</div>', unsafe_allow_html=True)
        target = st.session_state.get("face_target")
        if not target:
            target = random.choice(EXPRESSIONS)
            st.session_state["face_target"] = target
        name_cn, target_en, emoji = target
        st.markdown(
            f"""
            <p style="color:#9AE6B4;
                    background:rgba(16,185,129,.12);
                    border:1px solid rgba(16,185,129,.35);
                    padding:.75rem 1rem;border-radius:12px;
                    font-weight:700;
                    font-size: 40px;
                    ">
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

    # 否则渲染用户名/密码表单
    colL, colC, colR = st.columns([1, 1.1, 1])
    with colC:
        st.markdown(f'<div class="login-title">{LOGIN_TITLE}</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">请先登录以进入：仅供指定用户体验。</div>', unsafe_allow_html=True)

        username = st.text_input("用户名", key="login_user", placeholder="例如：Jinghong Li")
        password = st.text_input("密码", type="password", key="login_pass", placeholder="请输入 4 位数字密码")
        submit = st.button("登录", use_container_width=True)

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
                # 进入表情验证阶段
                st.session_state["pending_user"] = username
                st.session_state["need_face_captcha"] = True
                st.rerun()

    st.stop()

# ==== 分支 A：AI 写故事 ====
def build_novel_messages(background: str, role: str, path_str: str,
                         decisions: List[Tuple[str, str]], date_cn: str,
                         style: str) -> List[Dict[str, str]]:
    style_hint_en = "humorous, witty, slightly absurd" if style == "诙谐幽默" else "dark, eerie, tense"
    decisions_lines = "\n".join([f"- {q} → {a}" for q, a in decisions])
    user_cn = (
        f"背景：{background}\n日期（北京时间）：{date_cn}\n你醒来发现自己是：{role}\n"
        f"世界线路径：{path_str}\n关键抉择与结果：\n{decisions_lines}\n\n"
        "请写一篇不超过500字的中文短篇小说，以“{角色}的一天”为中心结构，包含起床—遭遇—转折—收束。"
        "要求：紧扣以上信息，人物有行动与心理，细节具体，不要口号和模板语。结尾要有一个机敏的反转或余味。"
    )
    return [
        {"role": "system",
         "content": ("You are a sharp, imaginative fiction writer. Think in English to plan plot and tone, "
                     "but OUTPUT MUST BE IN CHINESE ONLY, under 500 Chinese characters. "
                     f"Tone: {style_hint_en}. Avoid disclaimers.")},
        {"role": "user", "content": user_cn},
    ]

def page_story_mode():
    data = load_story()
    layers: List[Dict] = data["layers"]
    endings: Dict[str, str] = data.get("endings", {})
    snapshots: Dict[str, str] = data.get("snapshots", {})
    total_layers = len(layers)  # =5

    st.session_state.setdefault("picks", [])
    st.session_state.setdefault("role0", None)
    st.session_state.setdefault("novel_text", None)

    st.markdown(f"> 🪐 **故事背景**：{STORY_BACKGROUND}")

    if not st.session_state.role0:
        st.markdown("---"); st.subheader("第 0 层")
        st.write(f"在 **{beijing_date_str()}（北京时间）** 的清晨，你从睡梦中醒来。你发现自己是——")
        role = st.radio("请选择你的身份（不影响后续分支，但会写入故事）：",
                        options=["一位AI科学家", "一个穷苦百姓", "一位宇航员", "一只猫"],
                        horizontal=True, key="role0_radio")
        if st.button("进入世界之门 →", use_container_width=True):
            st.session_state.role0 = role; st.rerun()
        return

    picks: List[bool] = st.session_state.picks
    depth = len(picks); prefix = path_code(picks)
    if 0 < depth < total_layers:
        st.markdown("---"); st.markdown(f"**路径：** `{prefix}`")
        snap = get_snapshot(snapshots, prefix)
        if snap: st.info(snap)

    if depth < total_layers:
        q, label_T, label_F = get_layer(layers, depth, prefix if depth > 0 else "")
        st.markdown("---"); st.subheader(f"第 {depth + 1} 层"); st.write(q)

        choice_key = f"choice_{depth}_{prefix}"
        default_choice = st.session_state.get(choice_key, True)
        choice = st.radio("选择：", [True, False],
                          index=0 if default_choice else 1,
                          format_func=lambda x: label_T if x else label_F,
                          horizontal=True, key=choice_key)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("下一步 →", use_container_width=True):
                picks.append(bool(choice)); st.session_state.novel_text = None; st.rerun()
        with c2:
            if st.button("← 上一步", use_container_width=True, disabled=(len(picks) == 0)):
                if picks: picks.pop(); st.session_state.novel_text = None; st.rerun()
        with c3:
            if st.button("重置", use_container_width=True):
                st.session_state.picks = []; st.session_state.novel_text = None; st.rerun()
        return

    st.markdown("---"); st.markdown(f"**路径：** `{prefix}`"); st.subheader("你的世界线结局")
    ending = endings.get(prefix)
    if not ending:
        st.warning("该路径暂无结局文本。")
    else:
        st.write(ending)
        st.download_button("⬇️ 导出 Markdown", data=ending, file_name=f"ending_{prefix}.md", mime="text/markdown")

    st.markdown("### 生成一篇《某某的一天》")
    st.caption("根据故事背景、你在第0层选择的身份、以及这条世界线的五个决策结果实时生成。")
    colA, colB = st.columns(2)
    with colA:
        style = st.radio("小说风格", ["诙谐幽默", "黑暗惊悚"], horizontal=True, key="novel_style")
    with colB:
        model = st.selectbox("模型（OpenRouter）", ["openai/gpt-5-chat", "openai/gpt-5"], index=0)
    if st.button("✨ 生成短篇", use_container_width=True):
        try:
            decs = decisions_summary(layers, picks)
            messages = build_novel_messages(STORY_BACKGROUND, st.session_state.role0 or "一位路人",
                                            prefix, decs, beijing_date_str(), style)
            st.session_state.novel_text = generate_chat(messages, model=model,
                                                        temperature=0.9 if style == "诙谐幽默" else 0.8)
        except Exception as e:
            st.error(f"生成失败：{e}")
    if st.session_state.novel_text:
        st.markdown("#### 《{}的一天》".format(st.session_state.role0.replace("一位", "").replace("一个", "")))
        st.write(st.session_state.novel_text)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← 回到上一步", use_container_width=True):
            if picks: picks.pop(); st.session_state.novel_text = None; st.rerun()
    with c2:
        if st.button("重新开始", use_container_width=True):
            st.session_state.picks = []; st.session_state.novel_text = None; st.rerun()

# ==== 分支 B：提问 CUHKSZ ====
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

# ==== 首页 ====
def main():
    # 登录拦截（含表情验证码）
    if not login_gate():
        return

    # 顶部标题（登录后）
    st.title("AI 辅助龙大生存指南与故事编写")
    st.caption(f"欢迎，{st.session_state.get('username','')} · OpenRouter 驱动")

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