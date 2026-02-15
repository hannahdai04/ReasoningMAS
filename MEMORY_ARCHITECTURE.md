# Memory Architecture Documentation

本文档详细描述了 ReasoningMAS 项目中 AutoGen + HotpotQA 任务的 Private Memory 和 Shared Memory 架构。

---

## 1. 架构概览

系统采用**三层记忆架构**：

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
│    (Solver)      │  │   (Retriever)    │  │  (GroundTruth)   │
│  仅 Solver 可见   │  │ 仅 Retriever 可见 │  │ 仅 GroundTruth可见│
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## 2. Private Memory (Agent 私有记忆)

### 2.1 概述

- **文件位置**: `mas/agents/private_memory.py`
- **类名**: `AgentPrivateMemory`
- **生命周期**: 单次任务内（从 `reset()` 到任务结束）
- **可见性**: 仅创建它的 agent 可见

### 2.2 六大核心模块

#### 2.2.1 RoleContract (角色契约)

定义 agent 的身份和工作方式。

```python
@dataclass
class RoleContract:
    role_name: str              # 角色名称 (solver/retriever/verifier)
    responsibilities: List[str] # 职责清单
    boundaries: Dict[str, List[str]]  # {"can_do": [...], "cannot_do": [...]}
    priorities: List[str]       # 优先级策略 ["accuracy > speed", "safety > cost"]
    allowed_tools: List[str]    # 允许使用的工具/动作
    output_format: Optional[str] # 输出格式偏好
```

**示例 (Solver)**:
```python
solver.private_memory.set_role_contract(
    responsibilities=["分析问题并制定解决方案", "综合retriever的建议做出决策", "生成最终答案或动作"],
    boundaries={
        "can_do": ["推理", "选择动作", "使用retriever建议"],
        "cannot_do": ["直接访问ground truth", "跳过必要的检索步骤"]
    },
    priorities=["accuracy > speed", "evidence-based > speculation"],
    allowed_tools=["Think", "Search", "Lookup", "Finish"],
    output_format="Action[argument]"
)
```

基本是硬编码 

与 System Prompt 重复: autogen_prompt.py 中的 system prompt 已包含角色定义，RoleContract 成为冗余

#### 2.2.2 LocalPlan (本地任务计划)

管理 agent 的任务分解与待办事项。

```python
@dataclass
class TaskItem:
    task_id: str
    description: str
    status: str          # "pending" | "in_progress" | "completed" | "blocked"
    priority: int        # 数字越大越优先
    dependencies: List[str]
    next_action: Optional[str]
    created_at: str

@dataclass
class LocalPlan:
    tasks: Dict[str, TaskItem]
    current_focus: Optional[str]  # 当前聚焦的任务ID

  # 在 schedule() 中手动添加初始任务
  if solver.private_memory:
      solver.private_memory.add_task(
          task_id="main_task",
          description=task_main,  # 来自任务配置
          status="in_progress",
          priority=1,
          next_action="Analyze problem and plan solution"  # 硬编码
      )

  if retriever.private_memory:
      retriever.private_memory.add_task(
          task_id="retrieve_evidence",
          description="Suggest effective search/lookup actions",  # 硬编码
          status="pending",
          priority=1
      )
```

**API**:
- `add_task(task_id, description, status, priority, dependencies, next_action)`

- `update_task_status(task_id, status, next_action)`

- `get_pending_tasks()` → 按优先级排序的待处理任务

  status,piority没必要,next_action是硬编码

#### 2.2.3 HypothesesStore (假设存储)

管理推理过程中的假设和备选方案。

```python
@dataclass
class Hypothesis:
    hypo_id: str
    content: str           # 假设内容
    confidence: float      # 置信度 [0, 1]
    verified: bool         # 是否已验证
    pros: List[str]        # 优点
    cons: List[str]        # 缺点
    risks: List[str]       # 风险点
```

**API**:
- `add_hypothesis(hypo_id, content, confidence, pros, cons, risks)`

- `verify_hypothesis(hypo_id, verified)`

- `get_top_hypotheses(k)` → 置信度最高的 k 个假设

  

**容量控制**: 最多保留 `max_hypotheses` 个（默认10），按置信度淘汰。

#### 2.2.4 EvidenceTrace (证据与推理痕迹)

记录专业知识片段和推理过程。

```python
@dataclass
class EvidencePiece:
    evidence_id: str
    content: str           # 证据内容
    source: str            # 来源（observation/reasoning/external）
    relevance: float       # 相关度 [0, 1]
    supports: Optional[str] # 支持哪个假设ID
    timestamp: str

@dataclass
class ReasoningStep:
    step_id: str
    premise: str           # 前提
    conclusion: str        # 结论
    reasoning_type: str    # "deduction" | "induction" | "abduction"
    confidence: float
```

**API**:
- `add_evidence(evidence_id, content, source, relevance, supports)`
- `add_reasoning_step(step_id, premise, conclusion, reasoning_type, confidence)`
- `add_challenge(challenge)` → 记录对共享事实的质疑
- `get_recent_reasoning(k)` → 最近 k 步推理

**容量控制**: 最多保留 `max_evidence` 条（默认50），按相关度淘汰。

 Retriever     生成建议后      add_evidence()              f"Suggested: {retriever_action}"       只存建议，没存结果   │
  Solver 生成动作后add_reasoning_step()premise="Turn {i+1}: Based on current  模板化，无实际推理

 环境返回  observation 后    add_evidence()  observation[:200]



#### 2.2.5 PeerProfile (同伴画像)

维护对其他 agent 的轻量级画像。

```python
@dataclass
class PeerProfile:
    peer_name: str
    expertise: List[str]           # 擅长领域
    reliable_output_types: List[str]  # 可靠输出类型
    current_status: Optional[str]  # 当前状态描述
    interaction_count: int         # 交互次数
    success_rate: float            # 成功率（指数移动平均）
```

**API**:
- `update_peer_profile(peer_name, expertise, reliable_output_types, current_status)`
- `record_peer_interaction(peer_name, success)` → 更新成功率（EMA, α=0.3）硬编码
- 

#### 2.2.6 WorkingContext (工作上下文)

短期工作记忆，缓存最近的交互和状态。

```python
@dataclass
class InteractionRecord:
    turn: int
    peer_name: str
    message_type: str      # "request" | "response" | "suggestion"
    content: str
    timestamp: str

@dataclass
class WorkingContext:
    recent_interactions: List[InteractionRecord]  # 最近交互
    tool_results_cache: Dict[str, Any]            # 工具结果缓存
    key_observations: List[str]                   # 关键观察（最多20条）
    current_state_summary: Optional[str]          # 当前状态摘要
    max_interactions: int                         # 最多保留交互数（默认10）
```

**API**:
- `add_interaction(peer_name, message_type, content)`
- `cache_tool_result(tool_name, result)`
- `add_key_observation(observation)`
- `update_state_summary(summary)`
- `get_recent_interactions(k)` → 最近 k 轮交互
- 把日志存下来了,没有用

### 2.3 检索功能

支持两种检索模式：

#### 关键词检索（默认）
```python
results = private_memory.retrieve_relevant(
    current_query="Who directed Titanic?",
    current_state="",
    top_k=3,
    use_vector=False
)
```

#### 向量检索（需注入 embedding 函数）
```python
private_memory.set_embedding_func(embedding_func)
results = private_memory.retrieve_relevant(
    current_query="Who directed Titanic?",
    top_k=3,
    use_vector=True  # 自动检测
)
```

**返回结构**:
```python
{
    "relevant_reasoning": List[ReasoningStep],
    "relevant_hypotheses": List[Hypothesis],
    "relevant_evidence": List[EvidencePiece],
    "recent_interactions": List[InteractionRecord],
    "current_context": Optional[str]
}
```

### 2.4 Prompt 格式化

```python
prompt_str = private_memory.format_for_prompt(
    include_role_contract=True,
    include_plan=True,
    include_hypotheses=True,
    include_evidence=True,
    include_peers=False,
    include_context=True,
    compact=True,
    retrieved_data=retrieved_data  # 可选：使用检索数据
)
```

**输出示例**:
```
【Role】solver | Priorities: accuracy > speed | Tools: Search, Lookup, Finish

【Plan】
→ Analyze problem and plan solution [in_progress]
• Retrieve evidence [pending]

【Hypotheses】
~0.8 The director is James Cameron
~0.5 The director is Steven Spielberg ⚠ Less likely

【Reasoning】
∵ Titanic is a 1997 film ∴ Need to search for director
∵ Found James Cameron directed it ∴ Search for spouse

【Evidence】James Cameron directed Titanic (environment); Suzy Amis is spouse (search)

【Context】State: Turn 3: Progress | Recent: Found director | History: retriever→Search[...]
```

---

## 3. Shared Memory (共享记忆)

### 3.1 概述

- **文件位置**: `mas/agents/shared_memory.py`
- **类名**: `SharedMemory`
- **设计模式**: 单例模式
- **生命周期**: 单次任务内
- **可见性**: 所有 agent 可读可写

### 3.2 核心数据结构

#### 3.2.1 ReasoningHop (推理跳)

```python
@dataclass
class ReasoningHop:
    hop_id: int
    query: str                      # 本跳要回答的问题
    action: str                     # 执行的动作 (Search[...])
    observation: str                # 环境返回的结果
    extracted_answer: Optional[str] # 从 observation 中提取的答案
    agent: str                      # 执行的 agent (solver/retriever)
    timestamp: str
    success: bool                   # 本跳是否成功
```

#### 3.2.2 TaskStatus (任务状态)

```python
@dataclass
class TaskStatus:
    task_description: str
    current_hop: int
    total_hops_estimated: int       # 估计需要几跳（可动态调整）
    status: str                     # "in_progress" | "completed" | "failed"
    final_answer: Optional[str]
```

### 3.3 核心功能

#### 3.3.1 推理链管理

```python
# 添加推理跳
hop_id = shared_mem.add_hop(
    query="Who directed Titanic?",
    action="Search[Titanic director]",
    observation="James Cameron directed Titanic (1997).",
    extracted_answer="James Cameron",
    agent="solver",
    success=True
)

# 获取推理跳
last_hop = shared_mem.get_last_hop()
hop = shared_mem.get_hop(hop_id=1)
```

#### 3.3.2 检索去重

```python
# 检查是否已检索
if shared_mem.has_searched("Titanic director"):
    result = shared_mem.get_search_result("Titanic director")
else:
    # 执行检索...
    shared_mem.add_search_result("Titanic director", "James Cameron")
```

#### 3.3.3 中间事实累积

```python
shared_mem.add_fact("director", "James Cameron")
shared_mem.add_fact("spouse", "Suzy Amis")

director = shared_mem.get_fact("director")  # "James Cameron"
```

#### 3.3.4 Agent 交互记录

```python
# 记录 agent 间交互
shared_mem.record_interaction(
    from_agent="retriever",
    to_agent="solver",
    message="建议: Search[Titanic director]"
)

# 记录 agent 动作
shared_mem.add_agent_action(
    agent_name="solver",
    action_type="decision",
    content="Search[James Cameron spouse]",
    turn=2
)

# 记录协作结果
shared_mem.record_collaboration(
    agent1="retriever",
    agent2="solver",
    interaction_type="suggestion_accepted",
    outcome="success"
)
```

#### 3.3.5 任务状态管理

```python
shared_mem.reset(task_description="Who is the spouse of the director of Titanic?")
shared_mem.mark_completed(final_answer="Suzy Amis")
# 或
shared_mem.mark_failed()
```

### 3.4 检索功能

```python
results = shared_mem.retrieve_relevant(
    current_query="Who is James Cameron's spouse?",
    current_state="",
    top_k=3,
    use_vector=True  # 需先 set_embedding_func()
)
```

**返回结构**:
```python
{
    "relevant_hops": List[ReasoningHop],
    "relevant_facts": Dict[str, Any],
    "relevant_searches": Dict[str, str],
    "task_context": {
        "description": str,
        "current_hop": int,
        "status": str
    }
}
```

### 3.5 Prompt 格式化

```python
prompt_str = shared_mem.format_for_prompt(compact=True, retrieved_data=retrieved_data)
```

**输出示例**:
```
【共享记忆 - 所有Agent可见】
任务: Who is the spouse of the director of Titanic?
进度: Hop 2/2

推理链:
  Hop1 ✓: Search[Titanic director] → James Cameron directed Titanic (1997)
  Hop2 ✓: Search[James Cameron spouse] → James Cameron married Suzy Amis in 2000

已知事实:
  • director: James Cameron
  • spouse: Suzy Amis

已检索内容（避免重复）:
  • Titanic director
  • James Cameron spouse
```

### 3.6 统计信息

```python
stats = shared_mem.get_statistics()
# {
#     "total_hops": 2,
#     "successful_hops": 2,
#     "failed_hops": 0,
#     "unique_searches": 2,
#     "accumulated_facts": 2,
#     "agent_interactions": 4,
#     "task_status": "completed"
# }
```

---

## 4. Agent 响应流程 (Read → Inject → Act)

在 `mas/agents/base.py` 的 `Agent.response()` 方法中实现：

```
┌────────────────────────────────────────────────────────────────┐
│                         Agent.response()                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  (A) READ: 检索相关记忆                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. private_memory.retrieve_relevant(query, top_k=3)    │   │
│  │  2. SharedMemory().retrieve_relevant(query, top_k=3)    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  (B) INJECT: 格式化并注入到 prompt                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. shared_mem.format_for_prompt(retrieved_data)        │   │
│  │  2. private_memory.format_for_prompt(retrieved_data)    │   │
│  │  3. enhanced_prompt = memory_block + user_prompt        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  (C) ACT: 调用 LLM                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  messages = [                                            │   │
│  │      Message('system', system_instruction),              │   │
│  │      Message('user', enhanced_prompt)                    │   │
│  │  ]                                                       │   │
│  │  return self.reasoning(messages, reason_config)          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. AutoGen 工作流中的 Memory 使用

### 5.1 任务初始化

```python
# autogen.py: schedule() 方法

# 1. 初始化共享记忆
shared_mem = SharedMemory()
shared_mem.reset(task_description=task_main)

# 2. 重置所有 agent 的私有记忆
solver.private_memory.reset()
retriever.private_memory.reset()
ground_truth.private_memory.reset()

# 3. 多层记忆检索
# - Private Memory: agent 自身经验
# - Shared Memory: 团队共享知识
# - Meta Memory: 跨任务长期记忆
```

### 5.2 主循环中的 Memory 更新

```python
for turn in range(max_trials):
    # Retriever 动作后
    retriever.private_memory.add_evidence(...)
    retriever.private_memory.add_interaction(peer_name="solver", ...)
    shared_mem.add_agent_action(agent_name="retriever", action_type="suggestion", ...)

    # Solver 动作后
    solver.private_memory.add_reasoning_step(...)
    solver.private_memory.add_interaction(peer_name="retriever", ...)
    shared_mem.add_agent_action(agent_name="solver", action_type="decision", ...)

    # 环境反馈后
    solver.private_memory.add_key_observation(...)
    solver.private_memory.add_evidence(source="environment", ...)
    shared_mem.add_hop(query, action, observation, ...)

    # 协作记录
    shared_mem.record_collaboration(agent1="retriever", agent2="solver", outcome=...)
```

### 5.3 Solver 陷入循环时

```python
if solver_stuck:
    # 更新 solver 的私有记忆
    solver.private_memory.add_hypothesis(
        content="Current approach failed",
        confidence=0.0,
        risks=["Repeating same action leads to loop"]
    )
    solver.private_memory.add_challenge(f"Stuck at turn {i}")

    # 更新 ground_truth 的私有记忆
    ground_truth.private_memory.add_task(description="Break solver's loop", priority=10)
    ground_truth.private_memory.add_evidence(content="Solver stuck with repeated action")
    ground_truth.private_memory.update_peer_profile(
        peer_name="solver",
        current_status="stuck_in_loop"
    )
```

### 5.4 任务结束后

```python
# 更新私有记忆
if final_done:
    solver.private_memory.update_task_status("main_task", "completed")
else:
    solver.private_memory.add_challenge(f"Task failed: {feedback}")

# 更新共享记忆
if final_done:
    shared_mem.mark_completed(final_answer=feedback)
else:
    shared_mem.mark_failed()
```

---

## 6. HotpotQA 环境中的 Memory 应用

### 6.1 环境特点

- **任务类型**: 多跳问答 (Multi-hop QA)
- **动作类型**: `Search[query]`, `Lookup[term]`, `Finish[answer]`
- **典型流程**: 2-3 跳推理

### 6.2 Memory 如何辅助多跳推理

```
Question: Who is the spouse of the director of Titanic?

Hop 1:
├── Retriever 建议: Search[Titanic director]
├── Solver 执行: Search[Titanic director]
├── Observation: James Cameron directed Titanic (1997)
├── Private Memory:
│   └── add_evidence("James Cameron directed Titanic", source="environment")
└── Shared Memory:
    ├── add_hop(action="Search[Titanic director]", observation="James Cameron...")
    └── add_fact("director", "James Cameron")

Hop 2:
├── Retriever 建议: Search[James Cameron spouse]
├── Solver 执行: Search[James Cameron spouse]
├── Observation: James Cameron married Suzy Amis in 2000
├── Private Memory:
│   └── add_reasoning_step(premise="Director is James Cameron", conclusion="Search spouse")
└── Shared Memory:
    ├── add_hop(action="Search[James Cameron spouse]", observation="Suzy Amis...")
    └── add_fact("spouse", "Suzy Amis")

Final:
├── Solver 执行: Finish[Suzy Amis]
└── Shared Memory: mark_completed(final_answer="Suzy Amis")
```

### 6.3 检索去重

Shared Memory 防止重复检索：

```python
if shared_mem.has_searched("Titanic director"):
    # 直接使用缓存结果，避免重复 API 调用
    cached_result = shared_mem.get_search_result("Titanic director")
```

---

## 7. 向量检索配置

### 7.1 启用向量检索

```python
# autogen.py: build_system()
if config.get('use_vector_retrieval', True):
    embed_model = config.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2")
    embedding_func = EmbeddingFunc(embed_model)

    # 为所有 agent 的私有记忆注入
    for agent in [solver, retriever, ground_truth]:
        if agent.private_memory:
            agent.private_memory.set_embedding_func(embedding_func)

    # 为共享记忆注入
    SharedMemory().set_embedding_func(embedding_func)
```

### 7.2 相似度计算

```python
# mas/memory/utils.py
def cosine_similarity(vec1, vec2):
    # 计算两个向量的余弦相似度
    ...
```

---

## 8. 配置参数

在 `tasks/configs.yaml` 或运行时配置：

```yaml
# Private Memory 配置
max_interaction_history: 10    # 最多保留的交互记录数
max_evidence: 50               # 最多保留的证据数
max_hypotheses: 10             # 最多保留的假设数

# 向量检索配置
use_vector_retrieval: true
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

# 检索参数
successful_topk: 1             # 检索成功轨迹数
failed_topk: 1                 # 检索失败轨迹数
insights_topk: 3               # 检索 insights 数
threshold: 0                   # 相似度阈值
```

---

## 9. 总结

| 特性 | Private Memory | Shared Memory |
|------|----------------|---------------|
| **可见性** | 仅创建它的 agent | 所有 agent |
| **生命周期** | 单次任务 | 单次任务 |
| **设计模式** | 实例化 | 单例 |
| **核心功能** | 角色契约、任务计划、假设、证据、同伴画像、工作上下文 | 推理链、检索去重、事实累积、协作记录 |
| **检索方式** | 关键词/向量 | 关键词/向量 |
| **主要用途** | agent 内部推理优化 | 多 agent 协调同步 |

---

## 10. 文件索引

| 文件路径 | 说明 |
|----------|------|
| `mas/agents/private_memory.py` | Private Memory 实现 |
| `mas/agents/shared_memory.py` | Shared Memory 实现 |
| `mas/agents/base.py` | Agent 基类，实现 Read→Inject→Act |
| `tasks/mas_workflow/autogen/autogen.py` | AutoGen 工作流，Memory 调度逻辑 |
| `tasks/envs/hotpotqa_env.py` | HotpotQA 环境实现 |
| `mas/memory/utils.py` | 向量相似度等工具函数 |

---

## 11. 问题诊断与解决方案 (2026-02-04 分析)

### 11.1 当前问题总结

经过代码深入分析，发现 HotpotQA 任务成功率低的**根本原因**是：Memory 系统设计完善，但**几乎没有被正确使用**。

```
┌────────────────────────────────────────────────────────────────┐
│                     核心问题：数据流断裂                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LLM Response ──────────────────────────────────────────────X   │
│       │                                                         │
│       ▼                                                         │
│  env.process_action() ── 只提取 "Search[...]" 格式 ─────────→   │
│       │                                                         │
│       ▼                                                         │
│  env.step(action) ── 返回 observation ──────────────────────→   │
│       │                                                         │
│       ▼                                                         │
│  Memory Update ── 但存储的内容过于简单，关键信息丢失！ ───────X   │
│                                                                 │
│  ❌ extracted_answer 始终为 None                                │
│  ❌ 从 observation 中提取的实体/事实 从未存储                    │
│  ❌ add_fact() 从未被调用                                       │
│  ❌ has_searched() 去重机制从未使用                              │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

### 11.2 具体问题详解

#### 问题 1: `extracted_answer` 始终为 None

**位置**: `tasks/mas_workflow/autogen/autogen.py:518-525`

```python
# 当前代码
shared_mem.add_hop(
    query=f"Turn {i+1} execution",
    action=action,
    observation=observation,
    extracted_answer=None,  # ❌ 始终为 None!
    agent=name,
    success=(reward > 0)
)
```

**影响**:
- `SharedMemory.retrieve_relevant_insights()` 检查 `if hop.success and hop.extracted_answer:`
- 由于 `extracted_answer` 始终为 None，**insights 生成机制完全失效**
- 跨任务知识传递断裂

---

#### 问题 2: 从 Observation 中未提取关键实体

**现象**: 环境返回 `"James Cameron directed Titanic (1997)"` 但系统：
- 未提取 "James Cameron" 作为 bridging entity
- 未将其存入 `shared_mem.accumulated_facts`
- 下一跳无法利用已知信息

**需要添加**: Entity/Answer Extraction 模块

```python
# 应该有的逻辑
def extract_entity_from_observation(action: str, observation: str) -> Optional[str]:
    """从 Search/Lookup 结果中提取关键实体"""
    # 例如: Search[Titanic director] -> observation 包含 "James Cameron"
    # 应该提取 "James Cameron" 作为中间答案
    ...
```

---

#### 问题 3: `add_fact()` 从未被调用

**位置**: SharedMemory 有 `add_fact()` 方法，但 autogen.py 中**从未调用**

```python
# shared_memory.py 中定义了
def add_fact(self, key: str, value: Any):
    self.accumulated_facts[key] = value

# 但 autogen.py 中没有调用！
# 应该在获取 observation 后:
# shared_mem.add_fact("director", "James Cameron")
```

**影响**: 多跳推理时，第二跳无法知道第一跳发现了什么

---

#### 问题 4: 检索去重机制未使用

**位置**: SharedMemory 有 `has_searched()` / `add_search_result()` 但**从未调用**

```python
# 应该有的逻辑
if shared_mem.has_searched(query):
    cached = shared_mem.get_search_result(query)
    # 直接使用缓存，避免重复检索
else:
    # 执行检索
    shared_mem.add_search_result(query, result)
```

**影响**: Agent 可能重复检索相同内容，浪费 token 和步数

---

#### 问题 5: Private Memory 存储内容过于简单

**当前存储**:
```python
# autogen.py:294-299
retriever.private_memory.add_evidence(
    evidence_id=f"search_{i}",
    content=f"Suggested: {retriever_action}",  # ❌ 只存了建议，没存结果！
    source="self_generated",
    relevance=1.0
)
```

**应该存储**:
```python
# 应该存储完整的 action-observation 对
retriever.private_memory.add_evidence(
    evidence_id=f"search_{i}",
    content=f"Action: {action} -> Result: {observation[:200]}",
    source="environment",
    relevance=1.0,
    supports=hypothesis_id  # 关联到假设
)
```

---

#### 问题 6: LLM 响应未被解析利用

**现象**: Agent.response() 返回 LLM 的完整输出，但：
- HotpotQA 的 system prompt 要求只输出单行动作
- LLM 可能输出的 Thought/推理过程被 `process_action()` 丢弃
- 即使 LLM 有中间推理，也无法被 Memory 捕获

**AutoGen Prompt 设计**:
```python
# autogen_prompt.py
solver_system_prompt = """
...
Do not output thoughts, explanations, or multiple lines.  # ❌ 阻止了推理输出
...
"""
```

---

#### 问题 7: Memory 注入时机与内容不匹配

**当前流程**:
1. 检索 Private/Shared Memory
2. 格式化注入 prompt
3. 但注入的内容在第一轮几乎为空（因为还没执行任何动作）
4. 后续轮次虽然有内容，但**关键信息未被存储**

---

#### 问题 8: 检索 query 过长/噪声过高，导致检索命中率低

**位置**: `mas/agents/base.py` 的 `Agent.response()`
当前使用 `current_query=user_prompt` 进行检索，但 `user_prompt` 包含 few-shots、insights、历史轨迹等大段内容，导致：
- 关键实体/问题被噪声淹没
- 关键词检索与向量检索都容易偏离“当前子问题”

**影响**: Private/Shared Memory 检索结果不稳定且相关性弱，难以形成可复用上下文。

---

#### 问题 9: SharedMemory 的 hop 查询字段语义过弱

**位置**: `tasks/mas_workflow/autogen/autogen.py`
当前 `add_hop()` 使用 `query=f"Turn {i+1} execution"`，导致：

- hop 无法直接反映真实子问题
- 之后的检索难以命中与“桥接实体”相关的轨迹

**影响**: SharedMemory 的可检索性下降，insight 生成也更弱。

---

#### 问题 10: LLM 原始输出未记录，推理信息直接丢失

**现象**:
- `HotpotQAEnv.process_action()` 会截断输出，仅保留首行 action
- LLM 原始输出未保存到日志或 memory

**影响**:
- 任何隐含推理/中间结构化信息均不可见
- 你无法核查“LLM 实际返回内容”是否足够形成 memory

---

#### 问题 11: `get_recent_successes()` 未实现，导致私有检索恒为空

**位置**: `tasks/mas_workflow/autogen/autogen.py`
```python
if solver.private_memory and hasattr(solver.private_memory, 'get_recent_successes'):
    solver_private_insights = solver.private_memory.get_recent_successes(topk=2)
```
`AgentPrivateMemory` 未实现 `get_recent_successes()`，这一路始终为空。

---

#### 问题 12: Private/Shared Memory 为“单任务生命周期”，无法跨任务复用

**位置**: `autogen.py` 中每个任务开始时 `reset()`
**影响**:
- Private/Shared Memory 每次任务都清空，无法形成跨任务知识积累
- 真正跨任务复用只能依赖 Meta Memory (G-Memory)

---
### 11.3 解决方案

#### 方案 1: 添加 Entity Extraction 模块 (关键!)

```python
# 新增: mas/agents/entity_extractor.py

import re
from typing import Optional, List, Tuple

class EntityExtractor:
    """从 observation 中提取关键实体"""

    @staticmethod
    def extract_from_search(action: str, observation: str) -> Tuple[Optional[str], List[str]]:
        """
        从 Search 结果提取实体

        Returns:
            (primary_answer, list_of_entities)
        """
        # 解析 action 类型
        match = re.match(r'Search\[(.+)\]', action)
        if not match:
            return None, []

        query = match.group(1).lower()

        # 常见模式提取
        patterns = [
            # "X is a/an Y" pattern
            r'([A-Z][a-zA-Z\s]+?) (?:is|was) (?:a|an|the) (.+?)[\.,]',
            # "X directed/wrote/created Y" pattern
            r'([A-Z][a-zA-Z\s]+?) (?:directed|wrote|created|produced|founded) (.+?)[\.,]',
            # "Y was directed/written by X" pattern
            r'(.+?) (?:was|were) (?:directed|written|created|produced) by ([A-Z][a-zA-Z\s]+?)[\.,]',
            # "X married Y" pattern
            r'([A-Z][a-zA-Z\s]+?) married ([A-Z][a-zA-Z\s]+)',
            # "spouse of X is Y" pattern
            r'(?:spouse|wife|husband) (?:of|is) ([A-Z][a-zA-Z\s]+)',
        ]

        entities = []
        primary = None

        for pattern in patterns:
            matches = re.findall(pattern, observation, re.IGNORECASE)
            for m in matches:
                if isinstance(m, tuple):
                    entities.extend([e.strip() for e in m if e.strip()])
                else:
                    entities.append(m.strip())

        # 根据 query 确定 primary answer
        if 'director' in query and entities:
            primary = entities[0]
        elif 'spouse' in query or 'married' in query:
            primary = entities[-1] if entities else None
        elif entities:
            primary = entities[0]

        return primary, list(set(entities))

    @staticmethod
    def extract_from_lookup(observation: str) -> Optional[str]:
        """从 Lookup 结果提取答案"""
        # 移除 "(Result X/Y)" 前缀
        clean = re.sub(r'\(Result \d+/\d+\)\s*', '', observation)
        return clean.strip() if clean else None
```

---

#### 方案 2: 修改 autogen.py 主循环

```python
# 在 autogen.py 的主循环中添加

from mas.agents.entity_extractor import EntityExtractor

# 在 env.step() 之后添加:
observation, reward, done = env.step(action)

# ========== 新增：提取并存储关键实体 ==========
extracted_answer = None
extracted_entities = []

if action.startswith('Search['):
    extracted_answer, extracted_entities = EntityExtractor.extract_from_search(action, observation)
elif action.startswith('Lookup['):
    extracted_answer = EntityExtractor.extract_from_lookup(observation)

# 更新共享记忆：存储完整信息
shared_mem.add_hop(
    query=action,  # 使用真实子问题/动作作为 query
    action=action,
    observation=observation,
    extracted_answer=extracted_answer,  # ✅ 现在有值了
    agent=name,
    success=(reward >= 0 and 'Invalid' not in observation)
)

# 存储发现的事实
if extracted_answer:
    # 根据 query 类型确定 key
    if 'director' in action.lower():
        shared_mem.add_fact('director', extracted_answer)
    elif 'spouse' in action.lower() or 'married' in action.lower():
        shared_mem.add_fact('spouse', extracted_answer)
    elif 'nationality' in action.lower():
        shared_mem.add_fact('nationality', extracted_answer)
    else:
        shared_mem.add_fact(f'entity_{i}', extracted_answer)

# 记录检索历史（去重用）
if action.startswith('Search['):
    query_match = re.match(r'Search\[(.+)\]', action)
    if query_match:
        shared_mem.add_search_result(query_match.group(1), observation[:200])
```

---

#### 方案 3: 增强 Private Memory 存储

```python
# 修改 autogen.py 中的 Private Memory 更新逻辑

# 在 solver 动作后
if solver.private_memory:
    # 存储完整的 action-observation 对
    solver.private_memory.add_evidence(
        evidence_id=f"step_{i}",
        content=f"{action} → {observation[:150]}",
        source="environment",
        relevance=1.0 if reward >= 0 else 0.5
    )

    # 如果提取到了实体，存储为推理步骤
    if extracted_answer:
        solver.private_memory.add_reasoning_step(
            step_id=f"discovery_{i}",
            premise=f"From {action}",
            conclusion=f"Found: {extracted_answer}",
            reasoning_type="deduction",
            confidence=0.9
        )

        # 更新假设
        solver.private_memory.add_hypothesis(
            hypo_id=f"bridge_{i}",
            content=f"The answer involves {extracted_answer}",
            confidence=0.8
        )
```

---

#### 方案 4: 修改 System Prompt 以获取结构化输出 (可选)

**当前问题**: Prompt 要求 LLM 只输出单行，丢失了推理过程

**方案 A**: 保持单行输出，但在 Memory 中通过后处理提取信息

**方案 B**: 修改 prompt 要求结构化输出

```python
# autogen_prompt.py 修改建议

solver_system_prompt_v2 = """
You are the solver agent for HotpotQA-style multi-hop QA.

Output format (JSON):
{
  "thought": "brief reasoning (optional)",
  "action": "Search[query] | Lookup[keyword] | Finish[answer]",
  "extracted_info": "key entity found (if any)"
}

If you cannot use JSON, output: Action: <action>

Strategy:
- Parse the question and plan a 2-hop chain.
- After each Search/Lookup, identify the key entity that bridges to the next hop.
...
"""
```

---

#### 方案 5: 添加检索去重

```python
# 在 retriever 动作前检查

if retriever_action.startswith('Search['):
    query_match = re.match(r'Search\[(.+)\]', retriever_action)
    if query_match:
        query = query_match.group(1)
        if shared_mem.has_searched(query):
            # 跳过重复检索，使用缓存
            cached = shared_mem.get_search_result(query)
            self.notify_observers(f"[Dedup] Already searched '{query}', using cached result")
            retriever_action = ''  # 不再建议此动作
```

---


#### 方案 6: 缩短检索 query，避免噪声污染
```python
# 改用 task_main 或当前子问题作为检索 query
query = task_main  # 或从 action / retriever 建议中提取子问题
private_retrieved = private_memory.retrieve_relevant(current_query=query, top_k=3)
shared_retrieved = shared_mem.retrieve_relevant(current_query=query, top_k=3)
```
**收益**: 提高检索相关性，减少无关 few-shots/insights 的干扰。

---

#### 方案 7: 记录 LLM 原始输出（调试与可追溯）
```python
# 在 env.process_action() 之前记录 raw output
raw_output = solver.response(user_prompt, self.reasoning_config)
self.notify_observers(f"[LLM Raw] {raw_output[:300]}")
# 或存入 private_memory / prompt_logs
```
**收益**: 能直接验证 LLM 输出是否包含可沉淀的推理/结构化信息。

---

#### 方案 8: 实现 get_recent_successes() 或移除该分支
```python
# 在 AgentPrivateMemory 中新增
def get_recent_successes(self, topk=2):
    return [h.content for h in self.hypotheses_store.get_top_hypotheses(topk)]
```
或直接移除该调用，避免“以为空则误判”。

---

#### 方案 9: 让 Private/Shared Memory 可跨任务复用
- **方案 A**: 在任务结束时持久化保存（json/数据库），下个任务加载
- **方案 B**: 把高价值信息抽取成 insights，写入 Meta Memory

**收益**: 真正形成可复用经验，而非单任务短期缓存。

---
### 11.4 优先级排序
| 优先级 | 修改项 | 预期效果 |
|--------|--------|----------|
| **P0** | 添加 EntityExtractor | 解决 extracted_answer 始终为 None 的问题 |
| **P0** | 调用 add_fact() | 启用多跳推理的知识传递 |
| **P1** | 增强 Private Memory 存储 | 提供更丰富的上下文 |
| **P1** | 启用检索去重 | 避免浪费步数 |
| **P1** | 缩短检索 query | 提升检索相关性 |
| **P1** | 记录 LLM 原始输出 | 便于分析 prompt->response 是否有效 |
| **P2** | 修复 get_recent_successes | 避免“假空检索” |
| **P2** | 跨任务持久化 Private/Shared | 提升可复用性 |
| **P2** | 结构化输出 | 更好地捕获 LLM 推理过程 |

---

### 11.5 参考论文建议
根据检索到的相关工作，以下论文/方向可作为 Memory 设计的参考：
1. **MemGPT (arXiv:2310.08560)**: 将 LLM 记忆管理类比为“操作系统”，强调长短期记忆的可控调度。
2. **MemoryBank (arXiv:2305.10250)**: 引入长期记忆与时间衰减/更新机制，强调跨任务知识积累。
3. **A-MEM (arXiv:2502.12110)**: 自动化记忆优化与“存什么/何时更新”的策略学习。
4. **Generative Agents (arXiv:2304.03442)**: 通过记忆流 + 反思 + 计划，将经历转化为可复用高层策略。
5. **G-Memory (本项目)**: 现有三层图谱结构可作为“任务内 + 跨任务”融合的基线。

---

### 11.6 验证方法

修改后，检查以下指标：

```python
# 任务结束时打印
stats = shared_mem.get_statistics()
print(f"Total hops: {stats['total_hops']}")
print(f"Hops with extracted_answer: {sum(1 for h in shared_mem.reasoning_chain if h.extracted_answer)}")
print(f"Accumulated facts: {stats['accumulated_facts']}")  # 应该 > 0
print(f"Unique searches: {stats['unique_searches']}")

# 检查 insights 是否生成
insights = shared_mem.retrieve_relevant_insights(query="test", topk=5)
print(f"Generated insights: {len(insights)}")  # 修复后应该 > 0
```

---

*问题分析时间: 2026-02-04*
*预计修改文件: autogen.py, 新增 entity_extractor.py*

### 11.7 LLM-Aware Private Memory 优化方案（基于 10 篇论文）

#### 11.7.1 相关论文（10 篇）
1. **MemGPT (arXiv:2310.08560)**: 分层记忆与虚拟上下文管理，强调“何时写入/何时检索”。
2. **MemoryBank (arXiv:2305.10250)**: 长期记忆 + 遗忘曲线式更新，支持跨任务可持续记忆。
3. **Generative Agents (arXiv:2304.03442)**: 记忆流 → 反思 → 计划，将经历提炼为可复用策略。
4. **Reflexion (arXiv:2303.11366)**: 失败反思写入 episodic memory，显著提升后续表现。
5. **A-MEM (arXiv:2502.12110)**: Zettelkasten 风格结构化记忆 + 自动链接。
6. **RMM (arXiv:2503.08026)**: 前向/后向反思，动态组织记忆粒度与检索策略。
7. **MemGen (arXiv:2509.24704)**: 记忆触发器 + 记忆编织器，强调“何时激活记忆/如何融入推理”。
8. **Mem0 (arXiv:2504.19413)**: 抽取-整合-检索三阶段，并支持图式记忆。
9. **Voyager (arXiv:2305.16291)**: 技能库式长期记忆，复用动作模式。
10. **HELPER (arXiv:2310.15127)**: 语言-程序对的长期记忆检索，示例驱动执行。

---

#### 11.7.2 优化思路（从“事件驱动硬编码”到“LLM-Aware 记忆编排”）

当前 Private Memory 主要靠事件触发 + 硬编码模板写入，缺少“可解释抽取 + 反思压缩 + 动态组织”的能力。
建议引入一个**轻量 LLM 辅助的 Memory Curator**，只参与“写入与整理”，不改变主推理路径：

**(A) 结构化抽取层（Extract）**
- 触发点：每次 `action + observation` 后
- 产物：JSON 结构（实体、关系、证据、置信度、桥接实体、下一跳候选）
- 参考：A-MEM / Mem0

**(B) 反思压缩层（Reflect & Compress）**
- 触发点：每个 hop 或任务结束
- 产物：成功链路、失败原因、可复用检索策略
- 参考：Reflexion / Generative Agents

**(C) 组织链接层（Link & Organize）**
- 记忆条目增加 tags/keywords
- 建立“桥接实体 ↔ 相关问题 ↔ 证据”链接
- 参考：A-MEM / Mem0

**(D) 动态权重层（Score & Forget）**
- 重要性评分（task relevance + usage feedback）
- 时间衰减（遗忘曲线）
- 成功任务强化（reward-based update）
- 参考：MemoryBank / RMM

**(E) 触发与注入策略（Trigger & Inject）**
- 只有在“桥接实体缺失/检索重复/失败”时触发更强记忆注入
- 关键路径优先注入“反思/策略”，而非原始轨迹
- 参考：MemGPT / MemGen

**(F) 可复用动作库（Optional）**
- 把高频有效 Search/Lookup 模式沉淀为“检索技能”
- 任务开始时先检索相关技能
- 参考：Voyager / HELPER

---

#### 11.7.3 最小可落地版本（不改主推理链）
1. **新增 LLM-Aware Memory Curator**（仅写入，不改推理）
2. **抽取 JSON 结构**写入 Private/Shared Memory
3. **反思压缩**写入 Meta Memory 或 Shared Memory 的 insights
4. **动态权重**用于检索排序（替代纯关键词/向量相似度）

这样可以在保持现有 AutoGen 主流程不变的前提下，显著提升记忆可复用性与检索质量。

---

### 11.8 Memory 写入与检索计划（最小可落地版本）

本节给出一个“可直接落地”的最小版本，目标是让 Private/Shared Memory 真正可被检索与复用，同时不改主推理链路。

---

#### 11.8.1 写入计划（Write Plan）

**核心原则**：写入的是“结构化事实 + 关系 + 下一步”，而非流水账。

**写入触发点**：每次 `action + observation` 后。

**写入内容（最小集合）**
1. **Shared Memory**
   - `ReasoningHop`：必须包含 `query=action` 与 `extracted_answer`
   - `TaskStatus`：根据 Curator 的 `next_queries` 动态更新 `total_hops_estimated`
   - `Facts`：把 Curator 抽取的事实写入 `add_fact()`
2. **Private Memory**
   - `EvidenceTrace`：写入事实（fact）与关系（relation）
   - `Hypotheses`：写入候选答案/桥接实体
   - `LocalPlan`：把 `next_queries` 转成 pending 子任务

**最小写入规则**
- `facts` → Evidence + Shared Facts  
- `relations` → ReasoningStep  
- `answer` → Hop.extracted_answer + Hypothesis  
- `next_queries` → LocalPlan pending tasks  

---

#### 11.8.2 检索计划（Retrieve Plan）

**核心原则**：检索 query 必须聚焦“当前子问题 + 桥接实体”，避免用整段 prompt。

**Query 生成优先级**
1. **当前动作子问题**（来自 action）  
2. **Curator 抽取的桥接实体**  
3. **task_main**（兜底）  

**最终 query**：只保留 1~2 个关键字段（避免噪声）。

**检索目标**
1. **Private Memory**
   - Evidence: 相关事实  
   - Reasoning: 相关推理链  
   - Hypotheses: 候选答案  
   - LocalPlan: pending 下一跳  
2. **Shared Memory**
   - ReasoningHop: 上一步结论  
   - Facts: 已确认事实  
   - Search history: 去重  

**排序策略（最小可行）**
```
score = 0.6 * relevance(query, memory)
      + 0.2 * recency
      + 0.2 * success_rate
```

**注入策略**
- Shared Memory: Top‑3 hops/facts  
- Private Memory: Top‑3 evidence/reasoning/hypotheses  

这样可在保持 prompt 简洁的同时，确保推理连续性。

---

# prompt

 形成prompt的顺序:

1. system prompt:定义文件: tasks/mas_workflow/autogen/autogen_prompt.py, 使用文件: tasks/mas_workflow/autogen/autogen.py (在 build_system 方法中), 这里是solver,retriever,gt 三个身份应该如何做, 初始化时: 在 autogen.py 的 build_system() 方法中，创建 Agent 实例时传入. 调用时: 在 mas/agents/base.py 的 response() 方法中，作为消息列表的第一条消息 (Message('system', ...)) 发送给 LLM。

2. user prompt(G-memory构造,一个task内不变):

   * 构建逻辑: tasks/mas_workflow/autogen/autogen.py (在 schedule 循环内部)

     - 模板文件: tasks/mas_workflow/format.py (format_task_prompt_with_insights 函数)
     - 素材来源: tasks/prompts/hotpotqa_prompt.py (Few-shots)用example填进去
     - 包含内容
       这是 Agent 每一轮看到的"任务书"，由四个部分动态组装：Standard Few-shots: 固定的 2 个 HotpotQA 成功示例（教学）。Memory Few-shots: 从 Meta Memory 检索到的过去成功的类似任务轨迹（经验）。Insights: 从 G-Memory 检索到的通用规则/技巧（智慧）。Task Description: 当前任务的具体问题和状态摘要。Dynamic Suggestion: 如果 Retriever 刚才发话了，会追加一句 Retriever suggestion: Search[...] 每一轮 (Turn): 在 autogen.py 的 while 循环中，每次调用 agent.response() 之前动态生成。它作为基础文本传入 response()方法。

3. memory注入:发生在 Agent 准备回答的前一刻。每一轮都会变
   * 注入逻辑: mas/agents/base.py 的 response() 方法 (第 166-282 行)
   * 格式化逻辑:(format_for_prompt)
   * Shared Memory Block (共享记忆):内容: 团队当前的推理链 (Hop1, Hop2...)、已确认的事实、防止重复的搜索历史。
   * Private Memory Block (私有记忆):内容: 团队当前的推理链 (Hop1, Hop2...)、已确认的事实、防止重复的搜索历史。

     def response(self, user_prompt):
      # 1. 检索相关记忆
      shared_data = shared_memory.retrieve()
      private_data = private_memory.retrieve()
     
     # 2. 格式化成字符串块
      shared_block = shared_memory.format_for_prompt(shared_data)
      private_block = private_memory.format_for_prompt(private_data)
    
      # 3. 拼接到 User Prompt 前面！
      # 最终的 prompt = [共享记忆] + [私有记忆] + [分割线] + [User Prompt]
      enhanced_prompt = f"{shared_block}\n{private_block}\n{'='*60}\n{user_prompt}"
    
      # 4. 发送给 LLM
      messages = [
          {"role": "system", "content": self.system_prompt},
          {"role": "user", "content": enhanced_prompt}
      ]
      return llm.chat(messages)

```
LLM 最终看到的 Prompt 长这样

  [SYSTEM MESSAGE]
  You are the solver agent... Output Search/Lookup/Finish...

  [USER MESSAGE]
  【Shared Memory】
  - Hop 1: Search[Ed Wood] -> Found movie...
  - Facts: Ed Wood is a director...

  【Private Memory】
  - Role: Solver
  - Plan: Need to find nationality...
  - Evidence: Movie page doesn't have nationality...

  ============================================================

  ## Successful Examples
  (Few-shot 1...)
  (Few-shot 2...)

  ## Your Past Successes
  (Memory Trajectory 1...)

  ## Insights
  - "Use specific queries for people..."

  ## Your Turn
  Question: What is the nationality of...?
  Retriever suggestion: Search[Ed Wood (director)]

  这就是整个系统运作时 Prompt 的完整全貌。system 在最外层，user prompt 提供任务框架，而 memory 在最后一刻作为上下文注入到user prompt 的最前面。
```

注意:拆解了针对数据集hotpotqa prompt里的system prompt到autogen的autogen prompt角色prompt里,为了解决冲突.之后修改需要再修改回来.
