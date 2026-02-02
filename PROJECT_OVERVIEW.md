# G-Memory 项目介绍方案（含关键流程）

> 目标读者：希望快速理解 G-Memory 多代理系统架构、运行方式与关键执行流程的技术/研发同学。

## 1. 项目概述
本项目是 G-Memory（Tracing Hierarchical Memory for Multi-Agent Systems）的官方实现，核心价值是为多代理系统（MAS）提供**层级记忆结构**，让系统在跨任务的执行过程中持续积累可复用的经验与洞见，并在新任务到来时进行检索、融合与更新。  
项目面向多种任务环境（ALFWorld / PDDL / FEVER），支持多种 MAS 编排方式（AutoGen / MacNet / DyLAN），并可切换不同记忆机制（Empty / ChatDev / MetaGPT / Voyager / Generative / MemoryBank / G-Memory）。

## 2. 核心概念
### 2.1 多代理系统（MAS）
- `mas/mas.py` 定义统一调度接口 `MetaMAS`（build_system / schedule）。
- `tasks/mas_workflow/*` 提供三种 MAS 实现：
  - **AutoGen**：三代理（solver/retriever/ground_truth），带私有/共享记忆协作。
  - **MacNet**：图结构节点协作 + 决策节点汇总。
  - **DyLAN**：二维神经元网格，多轮多节点共识或排名裁决。

### 2.2 记忆体系（G-Memory）
`mas/memory/mas_memory/GMemory.py` 实现三层图谱：
- **Interaction Graph**：任务内轨迹与状态链（StateChain）。
- **Query Graph**：任务相似性图，用于检索相关任务。
- **Insight Graph**：抽取与维护可泛化的“规则/洞见”。

### 2.3 任务环境（Env）
`tasks/envs/*` 提供统一接口：`set_env()` / `step()` / `feedback()` / `process_action()`  
已集成：
- **ALFWorld** (`alfworld_env.py`)
- **FEVER** (`fever_env.py`)
- **PDDL** (`pddl_env/pddl_env.py`)

## 3. 目录结构速览
```
.
├─ tasks/                 # 任务入口、环境与 MAS workflow
│  ├─ run.py              # CLI 入口 + 任务主循环
│  ├─ configs.yaml        # 全局配置入口
│  ├─ envs/               # 环境实现与 Recorder
│  └─ mas_workflow/       # AutoGen / MacNet / DyLAN
├─ mas/                   # 核心 MAS 组件
│  ├─ mas.py              # MetaMAS 抽象接口
│  ├─ llm.py              # LLM 接口（OpenAI client）
│  ├─ reasoning/          # Reasoning 模块
│  └─ memory/             # MAS Memory + G-Memory
├─ data/                  # 数据集（ALFWorld / PDDL / FEVER）
├─ configs/               # LLM 配置（configs.yaml）
├─ logs/                  # 运行日志输出
└─ .db/                   # 记忆与索引持久化目录（运行后生成）
```

## 4. 运行准备与配置
- 安装依赖：`requirements.txt`
- 准备数据：放入 `data/alfworld`、`data/pddl`、`data/fever`
- 配置模型密钥：复制 `template.env` 为 `.env`，填入 `OPENAI_API_BASE` 与 `OPENAI_API_KEY`
- LLM 配置：`configs/configs.yaml`（max_token / temperature / num_comps 等）
- 任务与环境配置：`tasks/configs.yaml` 与 `tasks/env_configs/*_config.yaml`

## 5. 关键流程（从 CLI 到 Memory 更新）
### 5.1 主执行链路
入口文件：`tasks/run.py`
1. **读取配置**：加载 `tasks/configs.yaml` 与对应 `env_config`。  
2. **build_task**：创建 Env / Recorder / MAS Workflow / Task 列表。  
3. **build_mas**：实例化 Reasoning、Memory（含 Embedding），并构建 MAS 系统。  
4. **run_task**：逐任务执行：
   - `env.set_env(task_config)` 获得 `task_main` / `task_description`
   - 生成 few-shot + system prompt
   - `mas.schedule(task_config)` 执行核心策略
   - `recorder.task_end(...)` 写入指标

简化流程示意：
```
CLI
  -> tasks/run.py
    -> build_task()
    -> build_mas()
    -> run_task()
      -> mas.schedule()
        -> env.step()
        -> meta_memory.update()
      -> recorder.log()
```

### 5.2 MAS 内部执行（以 AutoGen 为例）
文件：`tasks/mas_workflow/autogen/autogen.py`
- 初始化三角色 agent + 私有记忆 + 共享记忆 + Meta Memory  
- 每轮执行：
  1) retriever 生成检索建议  
  2) solver 执行动作并与环境交互  
  3) 若陷入循环，ground_truth 干预  
  4) 将动作/观察写入 Meta Memory 与 Shared/Private Memory  
- 任务结束：调用 `meta_memory.save_task_context()` 与 `meta_memory.backward()`

### 5.3 Memory 检索与更新（G-Memory）
文件：`mas/memory/mas_memory/GMemory.py`
- **检索**：`retrieve_memory()`  
  - TaskLayer 图中找近邻任务  
  - Chroma 相似检索成功/失败轨迹  
  - InsightManager 取相关洞见  
- **更新**：`add_memory()`  
  - 轨迹稀疏化与清洗  
  - 写入 Chroma  
  - 触发洞见微调与合并（周期性）
- **反馈**：`backward()`  
  - 依据 reward 更新 insight score  

### 5.4 关键数据结构
`mas/memory/common.py`
- **MASMessage**：任务主描述 + 轨迹 + label + StateChain  
- **StateChain**：记录每轮 agent message 形成的有向图状态  

## 6. MAS Workflow 差异对比
### AutoGen
- 角色分工清晰（solver / retriever / ground_truth）
- 强调“私有记忆 + 共享记忆 + Meta Memory”协同  

### MacNet
- 图结构多 agent 节点 + 统一决策节点  
- Topology 由图掩码（graph mask）控制，按拓扑顺序执行  

### DyLAN
- 多列多行神经元网格  
- 支持“共识”或“排名裁决”式的收敛决策  

## 7. 评估与日志
- Recorder 位于 `tasks/envs/*`  
  - `AlfworldRecorder` / `FeverRecorder` / `PDDLRecorder`  
- 运行日志输出到：`logs/{task}/{mas_type}/{model}/total_task.log`  
- Token 统计：`mas/llm.py` + `tasks/run.py` 中 `get_price()`  

## 8. 产出与持久化（.db）
G-Memory 相关的持久化文件结构：
```
.db/{model}/{task}/{mas_type}/{memory}/
├─ task_layer_graph.pkl     # 任务图谱
├─ insights.json            # 洞见集合
└─ chroma/                  # 向量记忆库
```

## 9. 扩展点（如何二次开发）
- 新任务环境：实现 `BaseEnv` + `BaseRecorder` 并注册到 `tasks/envs/__init__.py`
- 新 MAS：实现 `MetaMAS.build_system` / `MetaMAS.schedule`，注册到 `tasks/mas_workflow/__init__.py`
- 新记忆模块：继承 `MASMemoryBase`，加入 `mas/module_map.py`
- 新推理方式：继承 `ReasoningBase` 并更新 `mas/reasoning`

## 10. 快速运行命令（示例）
```
./run_mas.sh

python tasks/run.py \
  --task alfworld \
  --reasoning io \
  --mas_memory g-memory \
  --max_trials 30 \
  --mas_type autogen \
  --model <your model>
```

---
如需更偏“架构图/流程图/面向产品介绍”的版本，可以在此文档基础上裁剪与重排。需要我继续细化哪一部分（如：G-Memory 机制、MacNet/DyLAN 对比、任务环境细节）请告诉我。  
