# LLM Prompt设计思路说明

## 两个LLM的分工

### Synthesizer LLM：主题综合器
**任务**：从原始chunks中提取和综合主题
**输入**：10个chunks + persona定义 + query
**输出**：主题数组（可能5-8个主题）

### LLM-Consolidator：洞察整理器
**任务**：从主题中选择最佳N个，转化为符合schema的insights
**输入**：Synthesizer的主题数组 + schema要求
**输出**：最终的insights.json（精确5个）

---

## 设计原理：模仿我的人工思考过程

### 我生成human baseline时的思维流程

#### 阶段1：理解和识别（对应Synthesizer）
1. **阅读chunks** → 理解每个chunk在讲什么
2. **识别主题** → 发现"AI风险"在多个chunk重复出现
3. **跨chunk连接** → chunk2和chunk3都讲风险，可以综合
4. **提取persona视角** → CIO关心治理、安全、TCO

#### 阶段2：选择和打磨（对应Consolidator）
5. **优先级排序** → 哪些主题对CIO最重要？
6. **写标题** → "Salesforce识别关键AI风险..."（不是crm-20250430）
7. **写摘要** → 综合信息+业务含义
8. **添加元数据** → cio_relevance字段，行动建议

---

## Synthesizer Prompt核心设计

### 设计目标：让LLM像人类分析师一样"阅读理解"

#### 1. 分步指导（Step 1-4）
```
Step 1: Read and Understand（理解每个chunk）
Step 2: Cross-Chunk Pattern Recognition（跨chunk找模式）
Step 3: Synthesize Themes（综合成主题）
Step 4: Extract Key Facts（提取关键事实）
```

**为什么这样设计**：
- 人类不会一次性处理所有信息
- 我自己也是先读→找模式→综合→提取事实
- 给LLM一个"思考顺序"

#### 2. 反例教学（❌ Bad vs ✅ Good）
```markdown
### ❌ Poor Synthesis
- Uses document IDs as titles (crm-20250430)
- Generic statements that could apply to any persona

### ✅ Good Synthesis
- Uses descriptive titles that communicate value
- Explicitly states "what this means for {persona}"
```

**为什么这样设计**：
- LLM常犯的错误就是用文档ID做标题
- 给出明确的反例可以避免这个问题
- 这是我在对比分析中发现的最大问题

#### 3. Persona视角强制
```json
"persona_relevance": {
  "keywords": ["<which persona keywords are relevant?>"],
  "impact_areas": ["<which impact areas affected?>"],
  "action_required": "<What should the persona do?>"
}
```

**为什么这样设计**：
- 我生成human insights时，每个都问自己"CIO为什么关心这个？"
- 强制LLM回答这3个问题，确保persona视角
- 这是解决"0个关键词"问题的关键

#### 4. 证据链要求
```json
"supporting_chunks": ["chunk_id_1", "chunk_id_2"],
"primary_source": {"chunk_id": "...", "source_type": "SEC 10-Q"}
```

**为什么这样设计**：
- 我写insights时每个声明都链接回chunk
- 这确保可追溯性（audit trail）
- 防止LLM"编造"信息

---

## Consolidator Prompt核心设计

### 设计目标：选择最优insights并符合schema

#### 1. 评分选择机制（替代规则拼接）
```
Selection criteria weights:
- Persona keyword coverage: 40%
- Source authority: 30%
- Recency (within 12 months): 20%
- Urgency: 10%
```

**为什么这样设计**：
- 我选择5个insights时也是这样权衡的
- SEC filing（高权威）+ CIO关键词多 → 优先
- 有明确的权重，LLM可以做trade-off

#### 2. Summary写作公式
```
[Sentence 1] State the key finding from authoritative source
[Sentence 2] Provide supporting details or context
[Sentence 3] Explain business/technical implications for persona
[Sentence 4, optional] Note urgency or timeline
```

**为什么这样设计**：
- 这就是我写summary的模板
- 给LLM一个"写作框架"比让它自由发挥更可控
- 确保每个summary都有：事实→细节→含义

#### 3. 置信度映射表
```
0.95: SEC filing or official financial document
0.90: Official press release or investor relations
0.85: Research report with data
...
```

**为什么这样设计**：
- 我设置confidence时就是这样判断的
- 明确的规则表，LLM可以直接查表
- 避免LLM随意给分数

#### 4. Schema验证清单
```markdown
Before outputting, verify each insight has:
- ✅ `id` is a valid chunk_id from supporting chunks
- ✅ `title` is descriptive (not a document ID)
- ✅ `summary` is 2-4 sentences
...
```

**为什么这样设计**：
- 我写完后也会检查一遍
- 给LLM一个"自查清单"
- 这是最后的质量保障

---

## 与我的思考过程的对应关系

| 我的人工步骤 | 对应的LLM | Prompt中的设计 |
|------------|----------|---------------|
| 读10个chunks理解内容 | Synthesizer | Step 1: Read and Understand |
| 发现"AI风险"主题重复 | Synthesizer | Step 2: Cross-Chunk Pattern Recognition |
| 综合chunk2+3成一个洞察 | Synthesizer | Step 3: Synthesize Themes |
| 问"CIO为什么关心这个？" | Synthesizer | persona_relevance字段要求 |
| 选择最重要的5个 | Consolidator | Step 1: Select Top N Insights (评分) |
| 写描述性标题 | Consolidator | Step 2.2: Craft Title (禁止文档ID) |
| 写2-4句summary | Consolidator | Step 2.3: Write Summary (公式) |
| 提取关键引用 | Consolidator | Step 2.4: Extract Evidence Snippet |
| 添加行动建议 | Consolidator | Step 2.6: cio_relevance.action_required |
| 最后检查格式 | Consolidator | Step 4: Validate Schema Compliance |

---

## 关键创新点

### 1. 两阶段分离
- **为什么不用一个LLM做完**？
  - 综合（创造性） vs 整理（结构化）是两种思维模式
  - 我自己也是先"想"再"写"
  - 分离后每个LLM任务更聚焦

### 2. Persona视角强制
- 不是"可选"的persona字段，而是**强制要求**回答：
  - "这对CIO意味着什么？"
  - "需要采取什么行动？"
- 这解决了"0个关键词"的根本问题

### 3. 反例驱动
- 不只告诉LLM"要做什么"
- 明确指出"系统当前的错误"（文档ID做标题）
- 用❌ Bad vs ✅ Good对比

### 4. 证据链追溯
- 每个insight都链接回chunk_id
- 每个summary要求包含specific facts
- Evidence snippet必须是verbatim引用

---

## 预期效果

使用这两个prompt后，系统应该能：

| 指标 | 当前系统 | 预期改进 |
|------|---------|---------|
| 标题质量 | 3/5是文档ID | 5/5描述性标题 |
| CIO关键词 | 0个 | 3-5个 |
| 跨chunk综合 | 0（简单列出） | 2-3个chunks合并成1个insight |
| persona_relevance字段 | 缺失 | 100%包含 |
| Gate-8 G8-04 | FAIL (0 < 2) | PASS (≥2) |

---

## 使用方法

### 1. Synthesizer调用
```python
synthesizer_prompt = load_template("tmp/synthesizer_llm_prompt.md")
filled_prompt = synthesizer_prompt.format(
    original_query=query,
    persona_role="CIO",
    persona_keywords=["governance", "security", "platform", "TCO", "data integration"],
    persona_impact_areas=["Risk Management", "Compliance", "Budget Planning"],
    num_chunks=len(chunks)
)
synthesized_themes = llm_call(filled_prompt, chunks_json)
```

### 2. Consolidator调用
```python
consolidator_prompt = load_template("tmp/consolidator_llm_prompt.md")
filled_prompt = consolidator_prompt.format(
    original_query=query,
    persona_role="CIO",
    persona_keywords=["governance", "security", "platform", "TCO", "data integration"],
    target_count=5,
    min_sources=3,
    min_recent=2
)
final_insights = llm_call(filled_prompt, synthesized_themes)
```

### 3. 替换现有的Consolidator
当前系统：
```python
# scripts/run_graph.py - Consolidator node
def consolidate_node(state):
    # 规则拼接：简单合并chunks
    return {"insights": simple_merge(chunks)}
```

改进后：
```python
def consolidate_node(state):
    # Step 1: Synthesizer LLM
    themes = synthesizer_llm(chunks, persona, query)

    # Step 2: LLM-Consolidator
    insights = consolidator_llm(themes, schema, persona)

    return {"insights": insights}
```

---

## 总结

这两个prompt的核心思想是：**把我的人工思考过程编码成LLM可执行的步骤**

- Synthesizer = 我读chunks和找模式的过程
- Consolidator = 我选择和打磨insights的过程
- Persona强制 = 我不断问"CIO为什么关心"的过程
- 证据链 = 我每个声明链接chunk的过程

如果LLM能执行这些步骤，理论上能接近human baseline的质量。
