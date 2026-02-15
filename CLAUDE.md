# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

G-Memory: A hierarchical memory architecture for multi-agent systems (MAS). The system captures generalizable insights and agent-specific collaboration trajectories across tasks using a graph-based design with three tiers: Insight Graph, Query Graph, and Interaction Graph.

## Running Experiments

```bash
# Setup environment
conda create -n GMemory python=3.12
conda activate GMemory
pip install -r requirements.txt

# Configure API keys (copy template.env to .env and fill in values)
cp template.env .env

# Run with shell script
./run_mas.sh

# Run specific task via command line
python tasks/run.py --task alfworld --reasoning io --mas_memory g-memory --max_trials 30 --mas_type autogen --model <model>
python tasks/run.py --task pddl --reasoning io --mas_memory g-memory --max_trials 30 --mas_type autogen --model <model>
python tasks/run.py --task fever --reasoning io --mas_memory g-memory --max_trials 15 --mas_type autogen --model <model>
```

## Key CLI Arguments

- `--task`: `alfworld`, `fever`, `pddl`
- `--mas_type`: `autogen`, `macnet`, `dylan`
- `--mas_memory`: `empty`, `chatdev`, `metagpt`, `voyager`, `generative`, `memorybank`, `g-memory`
- `--reasoning`: `io`
- `--max_trials`: max environment steps
- `--successful_topk`, `--failed_topk`, `--insights_topk`: memory retrieval parameters
- `--hop`: graph expansion hop count for task layer retrieval

## Architecture

### Core Components (`mas/`)

- **`mas.py`**: `MetaMAS` base class - unified management of agent teams, environment, and memory scheduling
- **`agents/base.py`**: `Agent` class with reasoning module and system instructions; `Env` base class for environments
- **`llm.py`**: LLM interface (`GPTChat`) wrapping OpenAI-compatible APIs
- **`module_map.py`**: Maps string identifiers to reasoning/memory module classes

### Memory System (`mas/memory/`)

- **`mas_memory/memory_base.py`**: `MASMemoryBase` - abstract base for all memory implementations with inside-trial (task context) and cross-trial (persistent) memory methods
- **`mas_memory/GMemory.py`**: G-Memory implementation with three layers:
  - `TaskLayer`: Graph of tasks with similarity-based edges, supports k-hop retrieval
  - `InsightsManager`: Manages insight rules with scoring, finetuning, and merging
  - Main memory uses ChromaDB for vector storage
- Other memory implementations: `voyager.py`, `memorybank.py`, `chatdev.py`, `generative.py`, `metagpt.py`

### MAS Workflows (`tasks/mas_workflow/`)

- **`autogen/`**: AutoGen-style multi-agent coordination
- **`macnet/`**: MacNet architecture
- **`dylan/`**: DyLAN (Dynamic LAN) with configurable node count and learning rate

### Task Environments (`tasks/envs/`)

- **`alfworld_env.py`**: ALFWorld household tasks
- **`fever_env.py`**: FEVER fact verification
- **`pddl_env/`**: PDDL planning domains (includes full pddlgym implementation)

### Entry Point (`tasks/run.py`)

`TaskManager` orchestrates experiment flow:
1. `build_task()`: Creates environment, recorder, and MAS workflow
2. `build_mas()`: Instantiates reasoning module, memory module, and builds agent system
3. `run_task()`: Iterates through tasks, manages agent scheduling and memory updates

## Data Organization

```
data/
├── alfworld/alfworld_tasks_suffix.json
├── pddl/test.json
└── fever/fever_dev.jsonl
```

Experiment outputs stored in `.db/<model>/<task>/<mas_type>/<memory_type>/`

## Configuration

- `tasks/configs.yaml`: Global settings, environment configs, MAS hyperparameters
- `tasks/env_configs/`: Per-environment YAML configs

## Deep Learning Vibe Coding Rules (Copy-Paste Ready

### 0. Core Principles (highest priority)

1. **Read before you write**: Before proposing any change, you must first locate the main training path: entry script → config loading → data loading → model forward → loss → backward/optimizer → eval/metrics → checkpointing.
2. **Do not break reproducibility**: No change may silently alter randomness or dataset splits. Preserve/update seed, deterministic settings, dataset versioning, and config logging.
3. **Prefer minimal changes**: Start with a **Minimal Viable Change (MVC/MVP)** that runs end-to-end; enhance incrementally. Avoid “big bang” refactors.
4. **Keep interfaces stable**: Unless explicitly allowed, do not change batch field names, tensor shape conventions, `forward()` output structure, checkpoint fields, or eval script inputs/outputs.
5. **One problem per change**: One PR/commit should address one purpose (e.g., add a loss OR modify dataloader). Do not mix unrelated refactors/formatting.

### 1. Required Outputs Before Changing Code (reading deliverables)

1. **System map (1 page)**: Bullet-point summary of key files, call chain, critical tensor shapes, where loss/metrics are computed, train/eval commands, and what is stored in checkpoints.
2. **Risk checklist**: At minimum, identify and verify:
   - Data leakage (train/val/test splits; augmentations crossing splits)
   - Label/target alignment (especially for sequences/detection/segmentation)
   - Metrics match the training objective (e.g., multiclass vs multilabel; macro vs micro)
   - Mixed precision / DDP consistency issues

### 2. Planning & Acceptance (no acceptance, no coding)

1. **Define acceptance criteria**: Every change must state how it will be proven correct, including at least:
   - A minimal end-to-end run (e.g., 50–200 steps)
   - Sanity checks for metric/loss ranges and trends
   - Shape/dtype/device checks on the critical path
2. **Milestone decomposition**: Break large changes into 2–5 small milestones. Each milestone must run independently and be easy to revert.

### 3. Rules While Modifying Code (tool must obey)

1. **Explicit change scope**: First declare the **files to modify** and the **files not to touch**. Keep the modified set as small as possible.
2. **Preserve backward compatibility**: If changes affect checkpoints/config:

- Provide backward-compatible loading logic (or a clear migration script)
- Clearly document how older checkpoints load and where errors would occur

1. **Add guardrails by default**: Add `assert`/checks on key inputs/outputs (shape, nan/inf, dtype, device), and make them toggleable (e.g., via a debug flag).
2. **No implicit behavior changes**: Never silently change learning rate, batch size, loss weights, augmentations, or evaluation logic. If necessary, make changes explicit in config and changelog.
3. **No unexplained perf/memory regressions**: If changes may affect performance:

- Explain expected impacts (ops/activations/caching/IO)
- Prefer adding simple profiling/logs (iter time, max memory)

### 4. Experiment & Comparison Rules (common DL failure points)

1. **Ablations must change one variable**: For comparisons, lock all other hyperparameters. Log commit, config, seed, dataset version, and metrics.
2. **Small run first, full run later**: Validate stability on small data/few steps before full training. Never jump straight to full runs as trial-and-error.
3. **Results must be explainable**: Improvements should have a plausible reason (loss down, less overfit, recall up). If not explainable, add diagnostics before more tuning.

### 5. Tool Output Format (deliver like an engineer)

1. **Every output must include**:

- Change summary (1–3 bullets)
- Modification list (by file/function)
- How to run/verify (commands + expected observations)
- Risks and rollback plan

1. **If uncertain, stop at proposal—don’t guess**: If key information is missing (batch schema, data field semantics, metric definition), provide 2 options + what must be confirmed. Do not invent assumptions and apply large edits.

