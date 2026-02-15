# Memory Architecture Documentation

AutoGen + FEVER Private Memory  Shared Memory 

要求:1.使用autogen,完成fever任务.agent role为:solver + ground_truth（见 tasks/mas_workflow/autogen/autogen.py）      - solver：主要执行 Search/Lookup/Finish      - ground_truth：solver 卡住时介入,有这两个角色,应该怎么设计memory字段

## 你的任务目标

你要在 AutoGen 多 agent（目前是 **solver + ground_truth** 两角色）框架里，构建一套 **LM-aware 的记忆机制**，让系统在做 **FEVER（或 HotpotQA 这类检索+推理任务）** 时：

1. **让 agent 自主决定存什么、何时存、何时取**（不是靠硬编码规则堆日志）。
2. **把协作从“对话消息传递”升级为“共同演化任务状态”**：所有关键进展能被恢复、复用、延续，减少重复检索与重复推理。
3. **显著提升端到端性能**（例如 FEVER score：标签正确且证据满足要求），同时让失败可复盘、可纠错、可迁移到下一样本。
4. **让 ground_truth 的介入变成可学习资产**：介入不仅解决当下卡点，还要沉淀成 solver 下次能直接使用的规则/技能/模板。

------

## 对 Private Memory 的要求（每个 agent 自己的“可复用工作经验”）

### 1) 面向“行为复用”

private memory 必须能帮助该 agent 下次更快完成同类步骤，至少覆盖：

- **检索轨迹复用**：有效 query 模板、失败 query 的负例约束、实体消歧策略
- **证据复用**：候选证据句与其立场判断（support/refute/neutral）
- **决策复用**：为什么判 SUPPORTED/REFUTED/NEI、缺口是什么

### 2) 面向“可诊断协作”

因为你只有 solver + ground_truth，private memory 还必须支持“可诊断交接”：

- solver 卡住时，能生成结构化的 **stuck_report**（尝试过什么、缺什么、当前假设、希望 ground_truth 介入的最小需求）
- ground_truth 的介入必须记录为 **intervention**（诊断+最小补丁+可执行指令+期望证据）

### 3) 面向“演化与遗忘”

private memory 不能无限增长、也不能只存原文：

- 需要 **importance / usage（被检索次数、最近使用）** 支持遗忘或降级
- 需要 **links（supports/refutes/fixes/derived_from）** 支持把经验串成“个人知识网”
- 需要 **consolidation（合并去重）**，避免同一策略/证据反复写入变噪声

> 对 FEVER 的强约束：任何“可作为证据”的私有记忆，必须可落地到 `wiki_title + sentence_id`，否则无法稳定复现与评估。

------

## 对 Shared Memory 的要求（跨 agent 的“任务状态与共识资产”）

你希望共享记忆不是聊天记录，而是“团队共同演化的任务状态”。因此 shared memory 应该满足：

### 1) 共享的对象必须是“可共识、可复用”的

优先共享三类“团队资产”：

- **Task-State Ledger（任务状态账本）**：关键状态迁移（pre_state → action → observation → post_state）
- **Canonical Evidence（规范证据）**：经过核验、可引用的证据集合（FEVER 的 evidence_set）
- **Canonical Facts/Slots（可复用事实槽）**：例如实体映射、关键关系、常用检索模板（尤其是经 ground_truth 介入确认后）

不建议共享的内容：

- solver 的原始中间推理长文本、未验证的猜测、噪声检索结果（除非标注为 tentative 且有用途）

### 2) Shared memory 必须支持“避免重复劳动”

共享层要能直接回答这些问题：

- “这条 claim/子命题已经检索过哪些页面/句子？”
- “哪些证据已经被验证可支持/反驳？”
- “目前卡在什么原因？缺什么证据？”
- “ground_truth 给过什么最小补丁？是否已经执行？”

---

## 1.结构

****

```
┌─────────────────────────────────────────────────────────────┐
│                      Meta Memory (跨任务)                    │
│  - 长期记忆，跨任务持久化                                      │
│  - 存储成功/失败轨迹、insights                                │
│  - 使用 ChromaDB 向量存储                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Shared Memory (任务内共享)                 │
│  - 单例模式，所有 agent 可读可写                               │
│  - 推理链追踪、检索去重、任务状态同步                           │
│  - 生命周期：单次任务                                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Private Memory  │  │  Private Memory  │  │  Private Memory  │

```

---

## 2. Private Memory (Agent 私有记忆)

### 2.1 模块与生命周期

- 位置: `mas/agents/private_memory.py`
- 类: `AgentPrivateMemory`
- 生命周期: 每个 agent 每个任务一份；在 schedule() 开始 reset()
- **solver 的 private memory**：记录“我做过什么、看到了什么、当前相信什么、为什么卡住”，强调**过程可复盘**与**下一步可执行**。

  **ground_truth 的 private memory**：记录“卡住诊断→最小干预→纠错规则→验证证据”，强调**可迁移的纠错模式**（下次 solver 直接用，不必再求助）。可以参考以下结果:

```
 `{
  "memory_id": "uuid",
  "task_id": "fever_claim_id",
  "role": "solver|ground_truth",
  "turn_id": 12,
  "created_at": "2026-02-05T12:34:56Z",

  "type": "query_attempt|candidate_evidence|evidence_set|verdict|stuck_report|intervention|reflection|skill|state_transition",
  "summary": "一句话摘要（检索入口）",
  "content": "可长文本/结构化内容",

  "entities": ["..."],
  "keywords": ["..."],
  "tags": ["dataset:fever", "stage:retrieve", "stuck:entity_ambiguity"],
  "importance": 0.0,

  "embedding_text": "用于向量化的文本（summary+关键字段拼接）",
  "links": [{"to_id": "memory_id_x", "relation": "supports|refutes|fixes|caused_by|derived_from"}],

  "usage": {"retrieved_count": 0, "last_accessed_at": null}
}` 
```



### 2.2 写入规则（当前实现）

写入发生在 `tasks/mas_workflow/autogen/autogen.py` 每回合 env.step() 之后。

请你思考得出一个在得到返回结果后,选取有效内容,更新private memory的方法

### 2.3 检索规则（当前实现）

检索入口: `mas/agents/base.py` 的 `Agent.response()`。

- 请你设计一个检索需要的private memory的方法,将当前有需要的private memory 拼到prompt里
- 

### 2.4 Prompt 注入

- Private Memory 在 Shared Memory 之后注入

---
## 3. Shared Memory (共享记忆)

### 3.1 模块与生命周期

- 位置: `mas/agents/shared_memory.py`
- 类: `SharedMemory`（全局单例）
- 生命周期: 每个任务一份；在 schedule() 开始 reset(task_description)
- 结构: reasoning_chain(ReasoningHop) / accumulated_facts / task_status / search_history / agent_interactions

### 3.2 写入规则（当前实现）

写入发生在 `tasks/mas_workflow/autogen/autogen.py` 每回合 env.step() 之后。

1) 请你思考得出一个在得到返回结果后,选取有效内容,更新 memory的方法
### 3.3 检索规则（当前实现）

- 请你设计一个检索需要的shared memory的方法,将当前有需要的memory 拼到prompt里
- 

### 3.4 Prompt 注入

- Shared Memory 先于 Private Memory 注入
- format_for_prompt(compact=True, retrieved_data=shared_retrieved)
- 注入内容: 任务上下文、相关 hops、已知事实、已检索内容（如果有）

---
## 4. Agent  (Read  Inject  Act)
 Agent  (Read  Inject  Act)

 `mas/agents/base.py`  `Agent.response()` 

```

                         Agent.response()                        

                                                                 
  (A) READ:                                           
     
    1. private_memory.retrieve_relevant(query, top_k=3)       
    2. SharedMemory().retrieve_relevant(query, top_k=3)       
     
                                                                
  (B) INJECT:  prompt                               
     
    1. shared_mem.format_for_prompt(retrieved_data)           
    2. private_memory.format_for_prompt(retrieved_data)       
    3. enhanced_prompt = memory_block + user_prompt           
     
                                                                
  (C) ACT:  LLM                                              
     
    messages = [                                               
        Message('system', system_instruction),                 
        Message('user', enhanced_prompt)                       
    ]                                                          
    return self.reasoning(messages, reason_config)             
     
                                                                 

```

---

## 5. AutoGen 工作流中的 Memory 使用

### 5.1任务初始化

```python
# autogen.py: schedule() 

# 1. 
shared_mem = SharedMemory()
shared_mem.reset(task_description=task_main)

# 2.  agent 
solver.private_memory.reset()
retriever.private_memory.reset()
ground_truth.private_memory.reset()

# 3. 
# - Private Memory: agent 
# - Shared Memory: 
# - Meta Memory: 
```

### 

---
