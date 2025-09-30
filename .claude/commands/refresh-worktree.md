---
name: refresh-worktree
description: 刷新当前 worktree 到最新 main
---

# 刷新 Worktree

当前 worktree 路径：$ARGUMENTS

1. 检查是否在 worktree 中：`git worktree list`
2. 如果是，执行重置流程
3. 如果不是，提示切换到正确的 worktree