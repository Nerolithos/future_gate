# -*- coding: utf-8 -*-
"""
FutureGate story mode: branching choices and short fiction generation.
Exposes page_story_mode() for Streamlit.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml
import streamlit as st

from ai_utils import generate_chat


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "story.yaml"


STORY_BACKGROUND = (
    "十年里，算法接管了生产排程与能源调度，淡水与稀土让旧秩序摇晃。"
    "穷人更忙，富人更孤独，城市像会呼吸的机器。有人把希望交给新技术，有人把灵魂交给旧信仰。"
    "你在一个普通清晨醒来，必须回答一串问题：我们把缰绳交给谁，怎样活下去？"
)


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
    return mapping[prefix]


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
    total_layers = len(layers)

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
