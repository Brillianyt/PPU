#!/usr/bin/env python3
"""从 Claude Code 的 JSONL 对话文件提取关键决策和洞察，输出 markdown"""
import json, sys
from pathlib import Path

src = Path("conversations/2026-08-14_session.jsonl")
out = Path("conversations/KEY_DECISIONS.md")

# 提取用户和助手消息（不含 tool_use / tool_result，纯文本）
entries = []
for line in src.read_text().splitlines():
    try:
        rec = json.loads(line)
    except Exception:
        continue
    msg = rec.get("message", {})
    role = msg.get("role", "")
    content = msg.get("content", [])
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = " ".join(
            c.get("text", "") for c in content if c.get("type") == "text"
        )
    else:
        text = ""
    text = text.strip()
    if not text or len(text) < 20:
        continue
    if role not in ("user", "assistant"):
        continue
    entries.append((role, text))

# 去重相邻重复（避免反复 push 同一段）
deduped = []
for role, text in entries:
    if deduped and deduped[-1] == (role, text):
        continue
    deduped.append((role, text))

with out.open("w") as f:
    f.write("# v1.2 关键决策与对话记录\n\n")
    f.write("> 从 Claude Code 对话 JSONL 自动提取，保留关键的技术决策、洞察、命令。\n\n")
    f.write("## 时间线（按对话顺序）\n\n")
    for i, (role, text) in enumerate(deduped[:200], 1):
        tag = "🙋" if role == "user" else "🤖"
        # 截断过长文本
        snippet = text[:600].replace("\n", " ")
        if len(text) > 600:
            snippet += "...(截断)"
        f.write(f"### {i}. {tag} {role}\n\n")
        f.write(f"{snippet}\n\n")
print(f"提取 {len(deduped)} 条 → {out}")
