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
