# System Map: G-Memory Multi-Agent System

This document traces the execution pipeline from CLI command through agent execution, memory updates, and evaluation.

## A) Quickstart Commands

**Training/Execution:**
```bash
# Full run via shell script
./run_mas.sh

# Direct Python command
python tasks/run.py --task alfworld --reasoning io --mas_memory g-memory --max_trials 30 --mas_type autogen --model <model>
```

**Available tasks:** `alfworld`, `fever`, `pddl`
**Available MAS types:** `autogen`, `macnet`, `dylan`
**Available memories:** `empty`, `chatdev`, `metagpt`, `voyager`, `generative`, `memorybank`, `g-memory`

**Config locations:**
- Global: `tasks/configs.yaml`
- Environment-specific: `tasks/env_configs/{alfworld,fever,pddl}_config.yaml`
- API keys: `.env` (copy from `template.env`)

---

## B) Backbone Call Chain

```
CLI Entry
└── tasks/run.py:117 → __main__
    ├── tasks/run.py:151 → build_task() → TaskManager
    ├── tasks/run.py:162 → build_mas() → initializes reasoning + memory modules
    └── tasks/run.py:163 → run_task()
        └── tasks/run.py:93 → for task_id, task_config in tasks:  # MAIN LOOP
            ├── tasks/run.py:96 → mas.env.set_env(task_config)
            ├── tasks/run.py:109 → mas.schedule(task_config)  # CORE EXECUTION
            │   └── [MAS-specific implementation]
            │       ├── autogen: tasks/mas_workflow/autogen/autogen.py
            │       ├── macnet: tasks/mas_workflow/macnet/graph_mas.py
            │       └── dylan: tasks/mas_workflow/dylan/dylan.py
            │           └── Agent.response() → reasoning(messages) → LLM call
            │           └── env.step(action) → (observation, reward, done)
            └── tasks/run.py:111 → recorder.task_end(reward, done)

LEARNING SIGNAL (equivalent to backward):
└── mas/memory/mas_memory/GMemory.py:293 → backward(reward)
    └── GMemory.py:296 → insights_layer.backward(insight, reward)
        └── GMemory.py:580 → inner_insight['score'] += reward
```

---

## C) Key Modules and Files

| Component | File | Class/Function |
|-----------|------|----------------|
| **Entry** | `tasks/run.py` | `__main__`, `run_task()` |
| **Config** | `tasks/configs.yaml` | YAML dict |
| **Dataset** | `tasks/envs/__init__.py` | `get_task()` |
| **Environment** | `tasks/envs/alfworld_env.py` | `AlfworldEnv` |
| | `tasks/envs/fever_env.py` | `FeverEnv` |
| | `tasks/envs/pddl_env/pddl_env.py` | `PDDLEnv` |
| **Agent** | `mas/agents/base.py` | `Agent` |
| **MAS Orchestration** | `mas/mas.py` | `MetaMAS` |
| **MAS Workflows** | `tasks/mas_workflow/autogen/autogen.py` | `AutoGen` |
| | `tasks/mas_workflow/macnet/graph_mas.py` | `MacNet` |
| | `tasks/mas_workflow/dylan/dylan.py` | `DyLAN` |
| **Reasoning** | `mas/reasoning/reasoning_modules.py` | `ReasoningIO` |
| **LLM Interface** | `mas/llm.py` | `GPTChat` |
| **Memory** | `mas/memory/mas_memory/GMemory.py` | `GMemory`, `TaskLayer`, `InsightsManager` |
| **Metrics/Recording** | `tasks/envs/base_env.py:32` | `BaseRecorder` |
| **Checkpointing** | `mas/memory/mas_memory/GMemory.py` | `_index_done()` methods |

---

## D) Data Contract (Message Flow)

**Task Config (input to schedule):**
```python
task_config: dict = {
    'task_main': str,           # Task identifier
    'task_description': str,    # Full task description
    'few_shots': list[str],     # Example trajectories
    # + environment-specific fields
}
```

**Agent Message Flow:**
```python
# Agent.response() input
messages: list[Message] = [
    Message('system', system_instruction),
    Message('user', user_prompt)
]

# LLM output
response: str  # Action text

# Environment step
env.step(action) → (observation: str, reward: float, done: bool)
```

**Memory Storage (MASMessage):**
```python
MASMessage = {
    'task_main': str,
    'task_description': str,
    'task_trajectory': str,      # Action-observation chain
    'label': bool,               # Success/failure
    'chain_of_states': StateChain,
    'extra_fields': dict         # 'key_steps', 'fail_reason', 'clean_traj'
}
```

---

## E) Checkpoint Contract

**What is saved:**

| Data | Format | Location | Save Function |
|------|--------|----------|---------------|
| Task memories | ChromaDB | `{working_dir}/{namespace}/` | `Chroma.add_documents()` |
| Task graph | Pickle | `{working_dir}/task_layer_graph.pkl` | `TaskLayer._index_done()` (line 457-460) |
| Insights | JSON | `{working_dir}/insights.json` | `InsightsManager._index_done()` (line 754-755) |

**Load on init:**
- ChromaDB: `GMemory.__post_init__()` line 38-41 (auto-loads from `persist_directory`)
- Task graph: `TaskLayer.__post_init__()` line 367-370 (`pickle.load()`)
- Insights: `InsightsManager.__post_init__()` line 478 (`load_json()`)

**Working directory pattern:** `.db/{model}/{task}/{mas_type}/{memory_type}/`

---

## F) Metric Location and Definition

| Environment | Recorder Class | Location | Metrics |
|-------------|---------------|----------|---------|
| **ALFWorld** | `AlfworldRecorder` | `alfworld_env.py:98-131` | Per-type success arrays, overall success rate |
| **FEVER** | `FeverRecorder` | `fever_env.py:122-144` | Total reward, success count, averages |
| **PDDL** | `PDDLRecorder` | `pddl_env.py:295-347` | Per-game-type dicts, constraint satisfaction |

**Reward computation:**

| Env | Success Condition | Reward Values |
|-----|-------------------|---------------|
| ALFWorld | `info['won'][0]` | 1 (win), 0 (valid), -1 (think/invalid) |
| FEVER | `match_exactly(answer, gold)` | 1 (correct), 0 (wrong), -1 (invalid) |
| PDDL | All goal literals satisfied | `satisfied_goals / total_goals` (0-1 metric) |

**Success rate formulas:**
- ALFWorld: `sum(results) / sum(counts)` (line 126)
- FEVER: `dones / counts` (line 143)
- PDDL: `sum(dones.values()) / sum(cnts.values())` (line 346-347)

**Logging format:** `'%(asctime)s - %(message)s'` to `{working_dir}/{namespace}.log`

---

## G) UNKNOWN / Needs Clarification

1. **MAS-specific schedule() implementation details** - Each MAS type (AutoGen, MacNet, DyLAN) implements its own agent coordination logic. Would need to read:
   - `tasks/mas_workflow/autogen/autogen.py`
   - `tasks/mas_workflow/macnet/graph_mas.py`
   - `tasks/mas_workflow/dylan/dylan.py`

2. **When exactly backward() is called** - Not visible in main loop. Likely called within MAS workflow after task completion. Need to trace `schedule()` implementations.

3. **Embedding model initialization** - Uses `sentence-transformers/all-MiniLM-L6-v2` via `EmbeddingFunc` in `mas/utils.py`, but exact loading mechanism unclear.
