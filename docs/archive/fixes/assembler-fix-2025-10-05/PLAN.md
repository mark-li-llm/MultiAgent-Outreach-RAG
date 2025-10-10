# Assembler 节点截断 Bug 修复计划（方案1）

## 📋 执行摘要

**问题**：Assembler 节点的强制 readability enforcement 破坏了 A2A 已验证合格的邮件内容。

**方案**：信任 A2A 的合规检查结果，只在必要时才执行截断，并增加防护逻辑避免"越截断越糟"。

**影响范围**：
- 修改文件：`scripts/run_graph.py` (1处，约30行代码)
- 风险等级：**低** (逻辑优化，不改变接口)
- 预计工时：1-2小时（含测试）

---

## 🔍 问题诊断总结

### 当前逻辑流程

```
Stylist (LLM) → 生成邮件 (108 words, grade 18.73)
       ↓
A2A Compliance → 检查 flags (结果: critical=[], warning=[])
       ↓
Assembler → 打包 + 强制截断
       ↓
最终输出 → 破损邮件 (48 words, grade 27.71) ❌
```

### 关键发现

1. **Word count**: 108 ✅ (< 160，符合要求)
2. **Readability grade**: 18.73 ❌ (> 10，触发截断)
3. **截断副作用**: Grade 从 18.73 → **27.71 ⬆️** (反而更糟)
4. **A2A 结果**: 已验证合格 (critical=[], warning=[])

### 根本原因

```python
# Line 715-742: run_graph.py
# 问题代码：无条件强制截断
iterations = 0
while (_grade(body) > 10 or _word_count(body) > 160) and iterations < 3:
    state["email_draft"]["body"] = _shorten_body(state["email_draft"]["body"])
    iterations += 1
```

**缺陷**：
1. ❌ 不检查 A2A 结果，盲目执行
2. ❌ 不检测截断是否有效（grade 反升）
3. ❌ 破坏性截断（语义不完整）

---

## 🎯 方案1：信任 A2A + 智能防护

### 核心原则

1. **信任链**：A2A 已经做了合规检查，Assembler 不应重复相同工作
2. **硬性要求优先**：Word count > 160 必须截断
3. **Grade 宽容**：允许适度超标，避免破坏性修正
4. **防护机制**：检测截断无效时立即停止

### 详细设计

#### **决策树**

```
开始
  ↓
检查 word_count
  ├─ > 160 → 强制截断（保留语义）
  └─ ≤ 160 → 检查 grade
              ├─ ≤ 15 → 不截断（信任 A2A）
              └─ > 15 → 检查 A2A 结果
                        ├─ critical=[] → 不截断（信任 A2A）
                        └─ critical≠[] → 尝试截断（带防护）
```

#### **新逻辑伪代码**

```python
# Step 1: 评估当前状态
body = state["email_draft"]["body"]
current_wc = _word_count(body)
current_grade = _grade(body)

# Step 2: Word count 硬性要求
if current_wc > 160:
    # 必须截断
    iterations = 0
    while current_wc > 160 and iterations < 3:
        body = _shorten_body(body)
        current_wc = _word_count(body)
        iterations += 1
    state["email_draft"]["body"] = body

# Step 3: Grade 宽容策略
elif current_grade > 15:  # 只处理严重超标（原来是 10，现在放宽到 15）
    # 检查 A2A 是否通过
    if compliance["flags"]["critical"] == []:
        # A2A 认为合格，信任 A2A，不截断
        pass
    else:
        # A2A 也有问题，尝试截断但要防护
        iterations = 0
        prev_grade = current_grade

        while current_grade > 10 and iterations < 3:
            new_body = _shorten_body(body)
            new_grade = _grade(new_body)

            # 防护：如果 grade 不降反升，停止截断
            if new_grade >= prev_grade:
                # 保留原内容，不再尝试
                break

            # 截断有效，应用
            body = new_body
            prev_grade = new_grade
            current_grade = new_grade
            iterations += 1

        state["email_draft"]["body"] = body
# else: grade ≤ 15，不处理
```

---

## 📝 实施步骤

### Phase 1: 备份与准备 (5分钟)

```bash
# 1. 创建备份
cp scripts/run_graph.py scripts/run_graph.py.backup_assembler_fix

# 2. 确认当前版本
git diff scripts/run_graph.py | head -20

# 3. 记录当前测试基线
cat outputs/test_fix2/email.json > /tmp/baseline_broken.json
```

### Phase 2: 代码修改 (15分钟)

**修改位置**：`scripts/run_graph.py` Line 715-742

**原代码**（删除）：
```python
# Final readability/length enforcement to satisfy Gate-6 thresholds
def _word_count(t: str) -> int:
    import re as _re
    return len(_re.findall(r"\b\w+\b", t or ""))
def _grade(t: str) -> float:
    import re as _re
    sentences = [s for s in _re.split(r"[.!?]+", t or "") if s.strip()]
    sents = max(1, len(sentences))
    words = max(1, _word_count(t))
    syllables = max(1, sum(len(_re.findall(r"[aeiouyAEIOUY]", w)) or 1 for w in _re.findall(r"\b\w+\b", t or "")))
    return 0.39 * (words / sents) + 11.8 * (syllables / words) - 15.59
def _shorten_body(b: str) -> str:
    # keep at most 3 bullets; limit bullets to 8 words, other lines to 10
    lines = b.splitlines()
    head = []
    bullets = []
    for ln in lines:
        if ln.strip().startswith("- "):
            bullets.append("- " + " ".join(ln.split()[1:9]))
        else:
            head.append(" ".join(ln.split()[:10]))
    bullets = bullets[:3]
    nb = "\n".join([ln for ln in head if ln.strip()] + bullets)
    return nb
iterations = 0
while (_grade(state["email_draft"]["body"]) > 10 or _word_count(state["email_draft"]["body"]) > 160) and iterations < 3:
    state["email_draft"]["body"] = _shorten_body(state["email_draft"]["body"])
    iterations += 1
```

**新代码**（替换）：
```python
# Final readability/length enforcement with A2A trust
def _word_count(t: str) -> int:
    import re as _re
    return len(_re.findall(r"\b\w+\b", t or ""))

def _grade(t: str) -> float:
    import re as _re
    sentences = [s for s in _re.split(r"[.!?]+", t or "") if s.strip()]
    sents = max(1, len(sentences))
    words = max(1, _word_count(t))
    syllables = max(1, sum(len(_re.findall(r"[aeiouyAEIOUY]", w)) or 1 for w in _re.findall(r"\b\w+\b", t or "")))
    return 0.39 * (words / sents) + 11.8 * (syllables / words) - 15.59

def _shorten_body(b: str) -> str:
    # keep at most 3 bullets; limit bullets to 8 words, other lines to 10
    lines = b.splitlines()
    head = []
    bullets = []
    for ln in lines:
        if ln.strip().startswith("- "):
            bullets.append("- " + " ".join(ln.split()[1:9]))
        else:
            head.append(" ".join(ln.split()[:10]))
    bullets = bullets[:3]
    nb = "\n".join([ln for ln in head if ln.strip()] + bullets)
    return nb

# Smart enforcement: trust A2A, only truncate when necessary
body = state["email_draft"]["body"]
current_wc = _word_count(body)
current_grade = _grade(body)

# Priority 1: Word count hard limit (must enforce)
if current_wc > 160:
    iterations = 0
    while current_wc > 160 and iterations < 3:
        body = _shorten_body(body)
        current_wc = _word_count(body)
        iterations += 1
    state["email_draft"]["body"] = body

# Priority 2: Readability grade (trust A2A if passed)
elif current_grade > 15:  # Only handle severe cases (relaxed from 10 to 15)
    # If A2A passed compliance, trust it even if grade is high
    if compliance["flags"]["critical"] == []:
        # A2A verified - trust it, no truncation
        pass
    else:
        # A2A also flagged issues - try truncation with safeguard
        iterations = 0
        prev_grade = current_grade

        while current_grade > 10 and iterations < 3:
            new_body = _shorten_body(body)
            new_grade = _grade(new_body)

            # Safeguard: stop if grade gets worse
            if new_grade >= prev_grade:
                # Truncation ineffective, keep original
                break

            # Apply effective truncation
            body = new_body
            prev_grade = new_grade
            current_grade = new_grade
            iterations += 1

        state["email_draft"]["body"] = body
# else: grade ≤ 15 and wc ≤ 160, no action needed
```

### Phase 3: 单元测试 (20分钟)

创建测试脚本验证修复逻辑：

```python
# scripts/test_assembler_fix.py
import re

def test_case_1():
    """Test: A2A passed, grade slightly high - should NOT truncate"""
    body = "Here's how recent momentum could support your CX outcomes..."  # 108 words, grade 18.73
    compliance = {"flags": {"critical": [], "warning": []}}

    # 运行新逻辑
    result = apply_fix(body, compliance)

    assert result == body, "Should preserve original when A2A passed"
    print("✅ Test 1 passed: A2A trust works")

def test_case_2():
    """Test: Word count > 160 - should truncate"""
    body = "Very long email..." * 20  # > 160 words
    compliance = {"flags": {"critical": [], "warning": []}}

    result = apply_fix(body, compliance)

    assert len(re.findall(r'\b\w+\b', result)) <= 160
    print("✅ Test 2 passed: Word count enforcement works")

def test_case_3():
    """Test: Grade worsens - should stop truncation"""
    # 模拟已经截断过一次的破损邮件
    body = "Here's how recent momentum could support your CX outcomes through"
    compliance = {"flags": {"critical": ["MISSING_FIELD"], "warning": []}}

    result = apply_fix(body, compliance)

    # 应该检测到 grade 升高，停止截断
    assert "through" in result  # 保留原文
    print("✅ Test 3 passed: Safeguard works")
```

### Phase 4: 集成测试 (15分钟)

```bash
# 1. 运行完整流程
OPENAI_API_KEY=... /Users/liyunxiao/anaconda3/bin/conda run -n age \
  python scripts/run_graph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id assembler_fix_test

# 2. 验证输出
cat outputs/assembler_fix_test/email.json | jq -r '.body' | wc -w
# 期望：100-110 words（完整）

cat outputs/assembler_fix_test/email.json | jq -r '.body'
# 期望：句子完整，无截断

# 3. 对比 A2A transcript
cat outputs/assembler_fix_test/a2a_transcript.jsonl | head -1 | jq -r '.content.body'
# 期望：与 email.json 中的 body 一致
```

### Phase 5: 回归测试 (10分钟)

验证不会破坏现有功能：

```bash
# 运行 Gate-8 evaluation
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  python scripts/qa_step08_generation_eval.py

# 检查是否仍然通过
cat reports/qa/step08_generation_eval.json | jq '.status'
# 期望：GREEN 或 AMBER（不应该是 RED）
```

---

## ✅ 验收标准

### 必须满足（P0）

- [ ] **完整性**：邮件正文句子完整，无截断痕迹
- [ ] **一致性**：`email.json` 与 A2A transcript 内容一致
- [ ] **Word count**：≤ 160 words（硬性要求）
- [ ] **回归**：Gate-8 evaluation 仍然通过

### 应该满足（P1）

- [ ] **可读性**：Grade ≤ 15（放宽后的标准）
- [ ] **性能**：截断循环 ≤ 3 iterations
- [ ] **日志**：清晰记录截断决策（可选）

### 良好实践（P2）

- [ ] **代码注释**：关键决策点有说明
- [ ] **测试覆盖**：3+ test cases
- [ ] **文档更新**：更新 CLAUDE.md 说明修复

---

## 🔄 回滚方案

### 如果修复失败

```bash
# 1. 立即回滚
cp scripts/run_graph.py.backup_assembler_fix scripts/run_graph.py

# 2. 验证回滚成功
git diff scripts/run_graph.py
# 应该无输出（恢复到原状）

# 3. 重新测试基线
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  python scripts/run_graph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id rollback_test

# 4. 分析失败原因
cat outputs/assembler_fix_test/email.json > /tmp/failed_fix.json
diff /tmp/baseline_broken.json /tmp/failed_fix.json
```

### 备选方案

如果方案1仍有问题，可以降级到更保守的方案：

```python
# 超简单版本：只在 word_count > 160 时截断
if _word_count(state["email_draft"]["body"]) > 160:
    # 仅处理超长邮件
    iterations = 0
    body = state["email_draft"]["body"]
    while _word_count(body) > 160 and iterations < 3:
        body = _shorten_body(body)
        iterations += 1
    state["email_draft"]["body"] = body
# 完全信任 A2A 对 readability 的判断
```

---

## 📊 预期效果

### 修复前 vs 修复后

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| **邮件完整性** | ❌ 破损（48 words） | ✅ 完整（103-108 words） |
| **语义连贯性** | ❌ 句子截断 | ✅ 句子完整 |
| **Word count** | ✅ 48 (< 160) | ✅ 103-108 (< 160) |
| **Grade** | ❌ 27.71 (截断后更糟) | ⚠️ 18.73 (超标但可接受) |
| **A2A 一致性** | ❌ 不一致 | ✅ 一致 |
| **用户体验** | ❌ 不可用 | ✅ 可用 |

### 性能影响

- **执行时间**：无影响（逻辑优化）
- **内存使用**：无影响
- **成功率**：预计提升（减少破坏性截断）

---

## 🎯 实施时间表

| 阶段 | 预计时间 | 负责人 | 检查点 |
|------|---------|--------|--------|
| 备份 | 5min | - | ✅ 备份文件存在 |
| 代码修改 | 15min | - | ✅ Diff review 通过 |
| 单元测试 | 20min | - | ✅ 3 test cases pass |
| 集成测试 | 15min | - | ✅ 完整邮件生成 |
| 回归测试 | 10min | - | ✅ Gate-8 仍通过 |
| 文档更新 | 5min | - | ✅ CLAUDE.md 更新 |
| **总计** | **70min** | | |

---

## 📌 关键决策记录

### 为什么选择方案1？

1. **最小改动**：只修改决策逻辑，不改变截断函数
2. **向后兼容**：保留所有现有功能，只加智能判断
3. **风险最低**：信任已有的 A2A 机制
4. **可逐步优化**：后续可迁移到方案2（智能截断）

### 关键假设

1. A2A 的 compliance check 是可靠的
2. Grade 15-20 的邮件在实际使用中是可接受的
3. Word count 是唯一的硬性要求

### 开放问题

- [ ] 是否需要记录截断决策的日志？
- [ ] 是否需要向用户暴露 grade 指标？
- [ ] 未来是否迁移到方案2（智能截断）？

---

## 🔗 相关文档

- Bug 分析：`/tmp/bug_analysis.md`
- 代码审查：`CODE_REVIEW_GATE8_DEBUG.md`
- 测试脚本：`scripts/test_llm_call.py`
- 备份位置：`scripts/run_graph.py.backup_assembler_fix`

---

**准备就绪，等待批准实施。**

*文档版本：1.0 - 2025-10-05*
