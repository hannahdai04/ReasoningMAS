# AutoGen 实现分析（本仓库）

说明：本文档基于本仓库代码实现与 AutoGen 论文（arXiv:2308.08155）对照撰写。**本仓库的 `autogen` 仅是一个 MAS 工作流实现，并非官方 AutoGen 库**。

## 1. 本仓库 AutoGen 的 agent 数量与角色（是否可编辑）
- **数量**：2 个。
- **角色**：
  - `solver`（解题主 agent）
  - `ground_truth agent`（在 solver 重复动作、判定卡住时介入）
- **证据位置**：
  - `tasks/mas_workflow/autogen/autogen.py` 的 `build_system()` 创建两个 `Agent`。
  - `tasks/mas_workflow/autogen/autogen_prompt.py` 定义角色 system prompts。
- **是否可编辑**：可以。
  - **改角色 / 提示词**：修改 `autogen_prompt.py` 的 `solver_system_prompt` / `ground_truth_system_prompt`。
  - **改数量 / 引入新角色**：在 `autogen.py` 的 `build_system()` 增加 `Agent` 并 `self.hire([...])`；同时在 `schedule()` 明确何时调用新 agent（否则不会被使用）。
  - **角色相关记忆投影**：若要新角色使用专属 insights，需同步调整 `_project_insights()` 或 `format_task_prompt_with_insights()`。

## 2. 本仓库 AutoGen 的 agent memory 设计
本仓库 AutoGen **没有独立 agent 记忆**，只有 **MAS 共享记忆（`meta_memory`）**。

### 2.1 Agent 级别
- `mas/agents/base.py` 的 `Agent` 支持 `memory_module` 参数，但 AutoGen 在 `build_system()` 里传的是 `None`。
- `Agent.response()` 仅基于 `system_instruction + user_prompt` 调用 LLM，不读写 agent 记忆。

### 2.2 MAS 级别（AutoGen 实际使用）
AutoGen 通过 `self.meta_memory` 操作 MAS 记忆：
- **试内记忆（inside-trial）**
  - `init_task_context()` 创建 `MASMessage`（含 `StateChain`）。
  - `add_agent_node()` 把 `AgentMessage` 作为图节点写入 `StateChain`。
  - `move_memory_state()` 记录 action / observation / reward，推进状态链。
- **跨任务记忆（cross-trials）**
  - 当 `--mas_memory g-memory` 时使用 `GMemory`：
    - **Chroma 向量库**存储任务轨迹。
    - **TaskLayer 图**：任务相似图（k-hop 扩展）。
    - **InsightsManager**：规则型 insights，支持打分、合并、微调。
  - `retrieve_memory()` 返回成功/失败轨迹 + insights，用于提示词拼接。
  - `project_insights()`（可选）把 raw insights 按 role 投影。
  - `backward()` 在任务结束后反馈奖励，更新 insights 分数。

### 2.3 记忆如何被注入提示词
- `tasks/mas_workflow/format.py` 的 `format_task_prompt_with_insights()` 组合：
  - dataset few-shots
  - 成功轨迹（memory few-shots）
  - insights
  - 当前任务描述
  形成最终 prompt。

## 3. 本仓库 AutoGen 的 agent 工作流程（schedule）
主要流程位于 `tasks/mas_workflow/autogen/autogen.py`：
1. **初始化**：读取 `task_config` → `env.reset()` → `meta_memory.init_task_context()`。
2. **检索记忆**：`meta_memory.retrieve_memory()` 获取成功轨迹 + insights。
3. **生成 prompt**：`format_task_prompt_with_insights()` 组合 few-shots、记忆、insights。
4. **主循环**（最多 `env.max_trials` 步）：
   - solver 生成 action。
   - 若 action 与最近 2 次重复 → 判定卡住 → 调用 ground_truth agent。
   - `meta_memory.add_agent_node()` 记录 agent 行为。
   - `env.step(action)` → observation/reward/done。
   - `meta_memory.move_memory_state()` 记录本步。
5. **收尾**：`env.feedback()` → `meta_memory.save_task_context()` → `meta_memory.backward()`。

## 4. 论文视角：AutoGen 的连接方式与消息传递模式（对照）
论文：`https://arxiv.org/abs/2308.08155`

### 4.1 连接方式（拓扑与对话模式）
- **核心范式**是“多智能体对话（multi-agent conversation）”。Agent 通过互相发送消息协作解决任务。
- **会话模式可静态或动态**：可预先规定流程，也可让对话内容驱动下一步交互。
- **动态群聊**：由管理者动态选择下一位发言者，并将其回复广播给所有成员。

### 4.2 消息传递与控制流
- **Conversable Agent**：具备统一的 send/receive 与生成回复能力，维护对话上下文；可由 LLM、工具、人类或混合方式驱动。
- **Auto-reply 机制**：默认一旦接收消息就自动调用 `generate_reply` 并把回复发回去，直到满足终止条件。
- **Conversation programming**：控制流由对话驱动，agent 的行为与消息传递共同构成流程；可通过自然语言或代码定义行为与终止条件。
- **动态扩展方式**：可在自定义 `generate_reply` 中触发与其他 agent 的对话，或用 function call 驱动新的 agent 参与。

## 5. 本仓库 AutoGen 的连接方式与消息传递模式
- **连接方式**：仅两个 agent，采取“solver 主导 + 卡住时切换 ground_truth” 的替补式连接；没有真正的 agent-to-agent 对话链路或群聊。
- **消息传递**：以 “agent → 环境 → 观测” 为主通道；agent 之间不直接互发消息，交互痕迹主要写入 `meta_memory` 的图结构。

## 6. 能否增加 agent 数量与补充 role？
- **论文/官方框架层面**：可以。AutoGen 设计就是定义多个 conversable agents，并通过对话驱动的控制流进行编排，支持自定义角色与动态对话模式。
- **本仓库实现层面**：可以，但需要改调度逻辑。
  - **新增 agent**：在 `autogen.py` 的 `build_system()` 创建并 `self.hire([...])`。
  - **接入流程**：在 `schedule()` 明确其调用条件或与其他 agent 的交互方式。
  - **提示词与角色**：在 `autogen_prompt.py` 增加新角色 system prompt；必要时扩展 `format_task_prompt_with_insights()`。
  - **若要模拟论文的动态群聊**：需要新增一个“管理者/调度器”逻辑，负责选择下一位发言者并广播消息（本仓库当前没有类似 GroupChatManager 的实现）。

## 7. 扩展 agent 自身 memory 或 agent 之间 memory 的切入点（本仓库）

### 7.1 给每个 agent 增加“个人记忆”
- **`mas/agents/base.py`**：
  - 扩展 `Agent.response()`：构建 messages 前 `self.memory.retrieve()`；获得 response 后 `self.memory.add()`。
  - 新增 agent 级 memory 抽象（如 `AgentMemoryBase`），或复用 `MASMemoryBase` 的接口但缩小范围。
- **`tasks/mas_workflow/autogen/autogen.py`**：
  - 创建 agent 时传入 `memory_module`。
  - 在 `schedule()` 中按角色注入不同记忆（或设置不同检索参数）。

### 7.2 增加“agent-之间”的交互记忆
- **`MASMessage` / `StateChain`**（`mas/memory/common.py`）：
  - `add_agent_node(agent_message, upstream_agent_ids)` 已支持图结构；当前 AutoGen 传空 `upstream_agent_ids`。
  - 可在 `schedule()` 中把当前 agent 节点与上一轮/其他 agent 节点连边，形成 inter-agent 关系图。
- **`GMemory`**：
  - 新增“agent message collection”或“interaction graph”索引。
  - 在 `retrieve_memory()` 中增加基于 agent-graph 的检索（如最近同角色/跨角色路径片段）。

### 7.3 把 inter-agent memory 融入提示词
- 在 `format_task_prompt_with_insights()` 增加“协作历史”段落（如“solver–ground_truth 协作轨迹”）。
- 或在 `_project_insights()` 中加入 agent 轨迹参数，使 insights 对特定 agent/对话更定制。

### 7.4 配置与模块注册
- 新 memory 类型需在 `mas/module_map.py` 注册。
- `tasks/run.py` 会把 `--mas_memory` 映射成实例，并把 `working_dir` / `hop` 等参数写入 `mem_config`。

---

## 8. 论文中的复杂推理任务：数据集 / 配置 / 角色 / 表现原因（对照）
> 参考：AutoGen 论文（arXiv:2308.08155）

### 8.1 复杂推理任务与数据集
- **A1 数学推理**：使用 **MATH** 数据集，评估在 120 个随机 level-5 题目和完整测试集上的表现（图示为 GPT-4）。  
- **A2 检索增强问答**：在 **Natural Questions** 数据集上评估 RAG 问答（图示为 GPT-3.5）。  
- **A3 交互式决策**：使用 **ALFWorld** 基准测试（文本世界环境任务）。  
- **A4 多智能体编码**：基于 **OptiGuide** 场景构造 100 个 coding 任务（安全/不安全各一半）。  
- **A5 动态群聊**：12 个手工设计的复杂任务的 pilot study（非标准基准）。  

### 8.2 论文中 AutoGen 的系统配置与角色
- **A1 数学推理（两 agent）**：论文描述为“直接复用两个内置 agent”的自动解题系统；人类参与时只需把 `human_input_mode='ALWAYS'` 打开。  
- **A2 RAG 问答与代码生成（两 agent）**：  
  - 角色：Retrieval-augmented User Proxy + Retrieval-augmented Assistant。  
  - 检索组件：**Chroma** 向量库 + **SentenceTransformers** 作为检索器。  
  - 机制：遇到检索不到信息时输出 “UPDATE CONTEXT” 触发继续检索。  
- **A3 ALFWorld（两或三 agent）**：  
  - 角色：Assistant（规划） + Executor（执行）。  
  - 增强：引入 **Grounding Agent** 注入常识性规则以避免错误循环；系统集成 ReAct 思路。  
- **A4 OptiGuide（多 agent）**：  
  - 角色：Commander（统筹）、Writer（写代码）、Safeguard（安全校验）。  
  - Commander 调用外部工具执行代码并回收结果，必要时把错误信息回传给 Writer 迭代。  
- **A5 Dynamic Group Chat（多 agent）**：  
  - 由 **GroupChatManager** 动态选 speaker 并广播消息。  
  - 论文使用 role-play 风格的 prompt 做 speaker selection。  

### 8.3 为什么在复杂推理任务上表现好（论文给出的关键因素）
- **多 agent 对话带来的能力互补**：论文强调多智能体对话可以利用反馈、组合不同能力并实现任务分解，从而帮助复杂推理。  
- **交互式检索与容错机制**：A2 中 “UPDATE CONTEXT” 的交互式检索显著提升检索增强问答的效果。  
- **引入 grounding agent 避免循环**：A3 中 grounding agent 在关键时刻注入常识约束，减少错误循环并提升成功率。  
- **安全与执行分工**：A4 中写代码与安全校验分离，并通过 Commander 统筹执行与复盘，使得安全识别 F1 显著提升。  
- **动态群聊的角色匹配**：A5 中基于角色扮演的 speaker-selection 提升成功率并减少调用次数。  

> 注：论文在多处将“详细评测与配置”放在 Appendix D（如 A1/A2 的具体设置与案例）。如果要把这些方案迁移到本仓库，可进一步对齐 Appendix D 的流程细节与提示词。  

---

## 快速定位文件（修改入口）
- AutoGen 工作流：`tasks/mas_workflow/autogen/autogen.py`
- AutoGen 提示词：`tasks/mas_workflow/autogen/autogen_prompt.py`
- 提示词拼接：`tasks/mas_workflow/format.py`
- Agent 类：`mas/agents/base.py`
- MAS 记忆基类：`mas/memory/mas_memory/memory_base.py`
- GMemory 实现：`mas/memory/mas_memory/GMemory.py`
- 模块映射：`mas/module_map.py`
- 运行入口与参数：`tasks/run.py`

# 数据集建议（按任务类型）



  - 多跳检索 / Evidence reasoning
      - HotpotQA：多跳问答 + 句级 supporting facts 标注，适合证据链评估。(hotpotqa.github.io (https://hotpotqa.github.io/index.html?utm_source=openai))
      - 2WikiMultiHopQA：提供 reasoning path/evidence 信息，专门面向多跳推理评估。(aclanthology.org (https://aclanthology.org/2020.coling-main.580/?utm_source=openai))
      - QASC：面向“句子组合”检索与推理，配 17M 句子语料库。(github.com (https://github.com/allenai/qasc?utm_source=openai))
      - MuSiQue：2–4 跳，多跳问题构造保证“connected reasoning”。(direct.mit.edu (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00475/110996/MuSiQue-Multihop-Questions-
        via-Single-hop-Question?utm_source=openai))
  - 逻辑 / 论证推理（Logical / Argument reasoning）
      - ReClor：逻辑推理阅读理解（考试题风格）。(microsoft.com (https://www.microsoft.com/en-us/research/publication/reclor-a-reading-comprehension-dataset-requiring-logical-
        reasoning/?utm_source=openai))
      - LogiQA：逻辑推理阅读理解数据集。(ijcai.org (https://www.ijcai.org/Proceedings/2020/501?utm_source=openai))
      - ARCT（SemEval-2018 Task 12）：给定 claim+premise+两条 warrant，选择正确 warrant。(aclanthology.org (https://aclanthology.org/S18-1121/?utm_source=openai))

  数据要求（请准备）

  - 需要把 HotpotQA dev JSON 放在：data/hotpotqa/hotpot_dev_distractor_v1.json
  - 代码假设每条样本包含字段：
      - question
      - answer
      - context（HotpotQA 官方格式：[title, [sentences]] 列表）
      - supporting_facts（已读入但当前不做评测）

  运行示例

  python tasks/run.py --task hotpotqa --mas_type autogen --mas_memory g-memory --reasoning io --model gpt-3.5-turbo-0125 --max_trials 20

Memory更新时机

  任务开始时

  - 所有agent的private memory自动reset
  - Solver添加主任务到计划
  - Retriever添加检索任务到计划

  每个Turn中

  Retriever执行后：
  - 记录检索动作为证据
  - 更新状态摘要
  - 记录与solver的交互
  - 更新solver的peer profile

  Solver执行后：
  - 记录推理步骤
  - 如果使用retriever建议，记录交互
  - 更新retriever的可靠性评分
  - 记录当前动作为推理结论

  Solver陷入循环时（Ground Truth介入）：
  - Solver记录失败假设和质疑点
  - Ground Truth记录介入任务和solver的失败模式
  - Ground Truth生成新动作后记录推理步骤

  环境反馈后：
  - 记录环境观察为证据
  - 根据reward更新状态摘要
  - 更新与retriever的协作成功率

  任务结束时

  - 根据成功/失败更新任务状态
  - 失败时记录失败原因到challenges

  使用方式

  方式1: 默认配置（已自动启用）

  python tasks/run.py --task alfworld --reasoning io --mas_memory g-memory --max_trials 30 --mas_type autogen --model <model>
  所有agent的私有memory已自动启用，无需额外配置！

  方式2: 自定义配置

  在 tasks/configs.yaml 中添加：
  autogen:
    max_interaction_history: 10  # 保留交互历史数
    max_evidence: 50            # 最大证据数
    max_hypotheses: 10          # 最大假设数

  方式3: 导出Memory用于分析（可选）

  在 autogen.py 的 schedule 方法末尾添加：
  # 保存所有agent的memory快照（调试用）
  import os
  debug_dir = os.path.join(self.meta_memory.persist_dir, "private_memory_logs")
  os.makedirs(debug_dir, exist_ok=True)

  if solver.private_memory:
      solver.private_memory.to_json(
          os.path.join(debug_dir, f"solver_task_{task_main[:20]}.json")
      )

  Turn i:
    1. Retriever生成建议 → 更新retriever memory
    2. Solver读取自己的memory（包含之前所有轮次的累积信息）
    3. Solver生成动作 → 更新solver memory
    4. 环境返回observation → 更新solver memory
    5. 如果stuck，Ground Truth介入 → 更新ground_truth memory