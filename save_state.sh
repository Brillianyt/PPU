#!/bin/bash
# save_state.sh — 一键保存当前实例的所有"易失"状态到 GitHub
# 
# 适用场景：
#   - pip install 新包后
#   - 跟 Claude Code 聊完一个重要话题
#   - 跑完一个重要实验
#   - 每天结束前
# 
# 做的事：
#   1. 更新 requirements-frozen.txt（pip freeze）
#   2. 复制最新的 Claude Code 对话 JSONL 到 conversations/
#   3. 重新生成 KEY_DECISIONS.md
#   4. git add + commit + push
#
# 前提：本地 git 已配置 + 远端认证可用

set -e
cd "$(dirname "$0")"

echo "===== 1) 更新 pip freeze ====="
pip freeze > environment_backup/requirements-frozen.txt
echo "  $(wc -l < environment_backup/requirements-frozen.txt) packages"

echo ""
echo "===== 2) 复制最新对话 JSONL ====="
LATEST_JSONL=$(ls -t /root/.claude/projects/-mnt-workspace-dndx-participant-v1-2/*.jsonl 2>/dev/null | head -1)
if [ -z "$LATEST_JSONL" ]; then
  echo "  ⚠ 找不到对话 JSONL（/root/.claude/projects/.../*.jsonl）"
else
  DATE=$(date +%Y-%m-%d)
  cp "$LATEST_JSONL" "conversations/${DATE}_session.jsonl"
  echo "  → conversations/${DATE}_session.jsonl ($(du -h "$LATEST_JSONL" | awk '{print $1}'))"
  
  echo ""
  echo "===== 3) 重新生成 KEY_DECISIONS.md ====="
  python3 conversations/extract_key_decisions.py 2>&1 || echo "  (跳过，extract 脚本可能不在)"
fi

echo ""
echo "===== 4) git status ====="
git status --short

echo ""
echo "===== 5) commit + push ====="
git add environment_backup/ conversations/
if git diff --cached --quiet; then
  echo "  无新内容，跳过 commit"
else
  MSG="state: $(date +%Y-%m-%d_%H:%M) 状态快照"
  git commit -m "$MSG" -m "包含：pip freeze $(wc -l < environment_backup/requirements-frozen.txt) 包 + 对话 JSONL"
  git push origin main
  echo "  ✓ push 完成"
fi

echo ""
echo "===== 完成 ====="
