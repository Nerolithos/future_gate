# future_gate
Online decision game that produces text by LLM



\# [L3] Future Gate

A/B 文字冒险 +（可选）AI 辅助建议，基于 Streamlit 1.47.0。

\# 本地运行

pip install -r requirements.txt

streamlit run app.py

模块结构：
- app.py：入口，选择模式并调用功能模块
- auth.py：登录与表情验证码
- futuregate.py：故事分支与小说生成
- cuhksz.py：CUHKSZ 问答
- ai_utils.py：OpenRouter 客户端与通用调用
