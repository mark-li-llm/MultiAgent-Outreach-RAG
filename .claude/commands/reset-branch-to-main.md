---
name: reset-branch-to-main
description: Reset branch to latest main
---

# 重置分支到最新 main

请帮我将当前分支重置为最新的 origin/main。

执行以下步骤：
1. 检查状态：`git status`
2. 获取最新：`git fetch origin main:main --force`
3. 重置分支：`git reset --hard origin/main`
4. 清理文件：`git clean -fd`
5. 推送更新：`git push origin HEAD --force-with-lease`
