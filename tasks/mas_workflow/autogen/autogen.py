from dataclasses import dataclass
import re
import json

from mas.agents import Agent
from mas.memory.common import MASMessage, AgentMessage
from mas.mas import MetaMAS
from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.memory import MASMemoryBase, GMemory
from mas.agents import Env
from mas.agents.shared_memory import SharedMemory
from mas.agents.memory_curator import MemoryCurator
from mas.utils import EmbeddingFunc

from .autogen_prompt import AUTOGEN_PROMPT
from ..format import format_task_prompt_with_insights, format_task_context


@dataclass
class AutoGen(MetaMAS):   

    def __post_init__(self):

        self.solver_name: str = 'solver'
        self.retriever_name: str = 'retriever'
        self.ground_truth_name: str = 'ground_truth'
        self.observers = []   

        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=['\n'])

    def build_system(self, reasoning: ReasoningBase, mas_memory: MASMemoryBase, env: Env, config: dict):

        self._successful_topk: int = config.get('successful_topk', 1)
        self._failed_topk: int = config.get('failed_topk', 1)
        self._insights_topk: int = config.get('insights_topk', 3)
        self._threshold: float = config.get('threshold', 0)
        self._use_projector: bool = config.get('use_projector', False)
        self.notify_observers(f"Successful Topk   : {self._successful_topk}")
        self.notify_observers(f"Failed Topk       : {self._failed_topk}")
        self.notify_observers(f"Insights Topk     : {self._insights_topk}")
        self.notify_observers(f"Retrieve Threshold: {self._threshold}")
        self.notify_observers(f"Use Role Projector: {self._use_projector}")

        if not isinstance(reasoning, ReasoningBase):
            raise TypeError("reasoning module must be an instance of ReasoningBase")
        if not isinstance(mas_memory, MASMemoryBase):
            raise TypeError("mas_memory module must be an instance of MASMemoryBase")

        # Private memory配置
        private_memory_config = {
            'max_interaction_history': config.get('max_interaction_history', 10),
            'max_evidence': config.get('max_evidence', 50),
            'max_hypotheses': config.get('max_hypotheses', 10)
        }

        # 所有agent都启用私有memory
        solver_agent: Agent = Agent(
            name=self.solver_name,
            role='solver',
            system_instruction=AUTOGEN_PROMPT.solver_system_prompt,
            reasoning_module=reasoning,
            memory_module=None,
            enable_private_memory=True,
            private_memory_config=private_memory_config
        )
        solver_agent.log_fn = self.notify_observers

        retriever_agent: Agent = Agent(
            name=self.retriever_name,
            role='retriever',
            system_instruction=AUTOGEN_PROMPT.retriever_system_prompt,
            reasoning_module=reasoning,
            memory_module=None,
            enable_private_memory=True,
            private_memory_config=private_memory_config
        )
        retriever_agent.log_fn = self.notify_observers

        ground_truth_agent: Agent = Agent(
            name=self.ground_truth_name,
            role="ground_truth",
            system_instruction=AUTOGEN_PROMPT.ground_truth_system_prompt,
            reasoning_module=reasoning,
            memory_module=None,
            enable_private_memory=True,
            private_memory_config=private_memory_config
        )
        ground_truth_agent.log_fn = self.notify_observers

        self._use_vector_retrieval: bool = config.get('use_vector_retrieval', True)
        if self._use_vector_retrieval:
            embed_model = config.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2")
            self._embedding_func = EmbeddingFunc(embed_model)
            for agent in [solver_agent, retriever_agent, ground_truth_agent]:
                if agent.private_memory:
                    agent.private_memory.set_embedding_func(self._embedding_func)
            SharedMemory().set_embedding_func(self._embedding_func)

        env_executor = env

        # Memory curator (write-only helper)
        llm_model = getattr(reasoning, "llm_model", None)
        self._memory_curator = MemoryCurator(llm_model=llm_model)

        self.hire([
            solver_agent,
            retriever_agent,
            ground_truth_agent
        ])
        self.set_env(env_executor)
        self.meta_memory = mas_memory

        # 为所有agent初始化角色契约
        self._initialize_agent_contracts(env)
        
    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, message: str):
        for observer in self.observers:
            observer.log(message)
    
    def schedule(self, task_config: dict) -> tuple[float, bool]:
        """
        Schedules and executes a task according to the given task configuration.
        This function initializes the task context based on the configuration, retrieves relevant memories and insights,
        and then executes the task by interacting with the environment and agents. It also handles memory updates and feedback.
        
        Parameters:
        - task_config (dict): A dictionary containing the task configuration, including the main task and description.
        
        Returns:
        - tuple[float, bool]: Returns the final reward and whether the task was successfully completed.
        """
        def _short(text: str, max_len: int = 160) -> str:
            if text is None:
                return ""
            text = str(text).replace("\n", " ").strip()
            return text if len(text) <= max_len else text[:max_len] + "..."

        if task_config.get('task_main') is None:
            raise ValueError("Missing required keys `task_main` in task_config")
        if task_config.get('task_description') is None:
            raise ValueError("Missing required keys `task_description` in task_config")
        
        task_main: str = task_config.get('task_main')
        task_description: str = task_config.get('task_description')
        few_shots: list[str] =  task_config.get("few_shots", [])
        
        # Initialize environment and agents
        env: Env = self.env
        solver: Agent = self.get_agent(self.solver_name)
        retriever: Agent = self.get_agent(self.retriever_name)
        ground_truth: Agent = self.get_agent(self.ground_truth_name)
        env.reset()

        self.meta_memory.init_task_context(task_main, task_description)

        # ========== 初始化共享记忆（新增）==========
        shared_mem = SharedMemory()
        shared_mem.reset(task_description=task_main)
        self.notify_observers("[Shared Memory] Initialized for new task")
        self.notify_observers(f"[Shared Memory] Global context: {shared_mem.get_global_context()[:100]}...")

        # 重置所有agent的私有memory（新任务开始）
        if solver.private_memory:
            solver.private_memory.reset()
        if retriever.private_memory:
            retriever.private_memory.reset()
        if ground_truth.private_memory:
            ground_truth.private_memory.reset()
        self.notify_observers("[Private Memory] Reset for all agents")

        # ========== 多层Memory检索（新增）==========
        # Step 1: 从Private Memory检索（agent自身的经验）
        solver_private_insights = []
        if solver.private_memory and hasattr(solver.private_memory, 'get_recent_successes'):
            solver_private_insights = solver.private_memory.get_recent_successes(topk=2)
            self.notify_observers(f"[Private Memory] Solver retrieved {len(solver_private_insights)} past successes")

        # Step 2: 从Shared Memory检索（团队共享的知识）
        shared_insights = shared_mem.retrieve_relevant_insights(query=task_main, topk=3)
        shared_collaboration_patterns = shared_mem.get_collaboration_history()
        self.notify_observers(f"[Shared Memory] Retrieved {len(shared_insights)} insights, {len(shared_collaboration_patterns)} collaboration patterns")

        # Step 3: 从Meta Memory检索（跨任务的长期记忆）
        successful_trajectories: list[MASMessage]
        failed_trajectories: list[MASMessage]
        insights: list[dict]

        successful_trajectories, failed_trajectories, insights = self.meta_memory.retrieve_memory(
            query_task=task_main,
            successful_topk=self._successful_topk,
            failed_topk=self._failed_topk,
            insight_topk=self._insights_topk,
            threshold=self._threshold
        )
        self.notify_observers(
            f"[Meta Memory] Retrieved {len(successful_trajectories)} success, "
            f"{len(failed_trajectories)} failed, {len(insights)} insights"
        )
        for idx, traj in enumerate(successful_trajectories):
            self.notify_observers(f"[Meta Memory][Success#{idx+1}] task_main={_short(traj.task_main, 100)}")
            key_steps = traj.get_extra_field('key_steps')
            if key_steps:
                self.notify_observers(f"[Meta Memory][Success#{idx+1}] key_steps={_short(key_steps, 200)}")
            self.notify_observers(f"[Meta Memory][Success#{idx+1}] traj={_short(traj.task_trajectory, 200)}")
        for idx, traj in enumerate(failed_trajectories):
            self.notify_observers(f"[Meta Memory][Failed#{idx+1}] task_main={_short(traj.task_main, 100)}")
            fail_reason = traj.get_extra_field('fail_reason')
            self.notify_observers(f"[Meta Memory][Failed#{idx+1}] fail_reason={_short(fail_reason, 200) or 'N/A'}")
        if insights:
            for idx, insight in enumerate(insights):
                self.notify_observers(f"[Meta Memory][Insight#{idx+1}] {_short(insight, 200)}")
        if failed_trajectories:
            self.notify_observers("[Usage] Failed trajectories retrieved but NOT injected into prompts (current behavior).")
        successful_shots: list[str] = [format_task_context(
            traj.task_description, traj.task_trajectory, traj.get_extra_field('key_steps')
        ) for traj in successful_trajectories]
        raw_rules: list[str] = [insight for insight in insights]
        roles_rules: dict[str, list[str]] = self._project_insights(raw_rules)
        if not self._use_projector:
            self.notify_observers("[Insights] Projector disabled (use_projector=False). Using raw insights for all roles.")
        elif not isinstance(self.meta_memory, GMemory):
            self.notify_observers("[Insights] Projector disabled (meta_memory is not GMemory). Using raw insights for all roles.")
        else:
            for role, rules in roles_rules.items():
                self.notify_observers(f"[Insights][{role}] projected_insights={len(rules)}")
        self.notify_observers(f"[Usage] Success cases injected into prompts: {len(successful_shots)}")
        self.notify_observers(f"[Usage] Insights injected into prompts: {len(raw_rules)}")

        # 为solver添加初始任务计划
        if solver.private_memory:
            solver.private_memory.add_task(
                task_id="main_task",
                description=task_main,
                status="in_progress",
                priority=1,
                next_action="Analyze problem and plan solution"
            )
            solver.private_memory.update_state_summary(f"Starting task: {task_main[:50]}")

        # 为retriever添加初始任务
        if retriever.private_memory:
            retriever.private_memory.add_task(
                task_id="retrieve_evidence",
                description="Suggest effective search/lookup actions",
                status="pending",
                priority=1
            )

        # Generate initial user prompt with insights
        user_prompt: str = format_task_prompt_with_insights(
            few_shots=few_shots,
            memory_few_shots=successful_shots,
            insights=raw_rules,
            task_description=self.meta_memory.summarize()
        )
        self.notify_observers(f"[Prompt] Initial prompt built with {len(successful_shots)} success cases and {len(raw_rules)} insights.")

        # Main loop for task execution
        action_history: list = [] 
        
        for i in range(env.max_trials):    
            
            retriever_prompt: str = format_task_prompt_with_insights(
                few_shots=few_shots, 
                memory_few_shots=successful_shots,
                insights=roles_rules.get(retriever.profile, raw_rules),
                task_description=self.meta_memory.summarize()
            )
            retriever_action: str = ''
            tries = 0
            while tries < 3:
                try:
                    retriever_action = retriever.response(retriever_prompt, self.reasoning_config)
                    if retriever_action == '':
                        continue
                    retriever_action = env.process_action(retriever_action)
                    break
                except Exception as e:
                    print(f'Error during execution of retriever agent: {e}')
                tries += 1

            if retriever_action.startswith('Finish['):
                retriever_action = ''

            if retriever_action != '':
                self.notify_observers(f"[Turn {i+1}] Retriever action: {_short(retriever_action, 120)}")
                retriever_message: AgentMessage = AgentMessage(
                    agent_name=retriever.name,
                    system_instruction=retriever.system_instruction,
                    user_instruction=retriever_prompt,
                    message=retriever_action,
                )
                self.meta_memory.add_agent_node(retriever_message, upstream_agent_ids=[])

                # 更新retriever的私有memory
                if retriever.private_memory:
                    # 记录检索动作为证据
                    retriever.private_memory.add_evidence(
                        evidence_id=f"search_{i}",
                        content=f"Suggested: {retriever_action}",
                        source="self_generated",
                        relevance=1.0
                    )
                    # 更新状态
                    retriever.private_memory.update_state_summary(f"Turn {i+1}: Suggested {retriever_action[:30]}")
                    # 记录与solver的交互
                    retriever.private_memory.add_interaction(
                        peer_name=solver.name,
                        message_type="suggestion",
                        content=retriever_action[:100]
                    )
                    # 更新peer profile
                    retriever.private_memory.update_peer_profile(
                        peer_name=solver.name,
                        current_status="waiting_for_action"
                    )

                # 更新共享记忆：记录retriever的建议
                shared_mem.add_agent_action(
                    agent_name=retriever.name,
                    action_type="suggestion",
                    content=retriever_action,
                    turn=i
                )
                self.notify_observers(f"[Shared Memory] Turn {i+1}: Retriever suggested action")
            else:
                self.notify_observers(f"[Turn {i+1}] Retriever action: (empty)")

            user_prompt: str = format_task_prompt_with_insights(
                few_shots=few_shots,
                memory_few_shots=successful_shots,
                insights=roles_rules.get(solver.profile, raw_rules),
                task_description=self.meta_memory.summarize()
            )
            if retriever_action != '':
                user_prompt = f"{user_prompt}\n\nRetriever suggestion: {retriever_action}"
            tries = 0

            while tries < 3:
                try:
                    action: str = solver.response(user_prompt, self.reasoning_config)
                    if action == '':
                        continue
                    action = env.process_action(action)
                    break
                except Exception as e:
                    print(f'Error during execution of solver agent: {e}')
                tries += 1

            # 更新solver的私有memory（动作生成后）
            if solver.private_memory:
                # 记录当前动作为推理步骤
                solver.private_memory.add_reasoning_step(
                    step_id=f"step_{i}",
                    premise=f"Turn {i+1}: Based on current state and retriever suggestion",
                    conclusion=f"Decided to execute: {action}",
                    reasoning_type="deduction",
                    confidence=0.8
                )
                # 如果有retriever建议，记录交互
                if retriever_action != '':
                    solver.private_memory.add_interaction(
                        peer_name=retriever.name,
                        message_type="suggestion",
                        content=retriever_action[:100]
                    )
                    # 更新retriever的可靠性
                    solver.private_memory.update_peer_profile(
                        peer_name=retriever.name,
                        expertise=["search", "lookup"],
                        reliable_output_types=["retrieval_suggestions"],
                        current_status="active"
                    )

            # 更新共享记忆：记录solver的决策
            shared_mem.add_agent_action(
                agent_name=solver.name,
                action_type="decision",
                content=action,
                turn=i
            )
            # 如果使用了retriever的建议，记录协作
            if retriever_action != '':
                shared_mem.record_collaboration(
                    agent1=retriever.name,
                    agent2=solver.name,
                    interaction_type="suggestion_accepted",
                    outcome="pending"
                )
                self.notify_observers(f"[Shared Memory] Turn {i+1}: Recorded Retriever→Solver collaboration")

            name: str = solver.name
            system_instruction = solver.system_instruction

            if self._solver_stuck(action, action_history):
                self.notify_observers(
                    f"[Turn {i+1}] Solver stuck on repeated action: {_short(action, 120)}. GroundTruth will intervene."
                )
                # 更新solver的私有memory：记录陷入循环
                if solver.private_memory:
                    solver.private_memory.add_hypothesis(
                        hypo_id=f"stuck_{i}",
                        content=f"Current approach failed: {action}",
                        confidence=0.0,
                        risks=["Repeating same action leads to loop"]
                    )
                    solver.private_memory.add_challenge(f"Stuck at turn {i+1}: {action}")

                # 更新ground_truth的私有memory：记录介入原因
                if ground_truth.private_memory:
                    ground_truth.private_memory.add_task(
                        task_id=f"intervene_{i}",
                        description=f"Break solver's loop at turn {i+1}",
                        status="in_progress",
                        priority=10
                    )
                    ground_truth.private_memory.add_evidence(
                        evidence_id=f"solver_loop_{i}",
                        content=f"Solver stuck with repeated action: {action}",
                        source="observation",
                        relevance=1.0
                    )
                    # 记录solver的失败模式
                    ground_truth.private_memory.update_peer_profile(
                        peer_name=solver.name,
                        expertise=["problem_solving"],
                        current_status=f"stuck_in_loop_{action[:20]}"
                    )

                user_prompt: str = format_task_prompt_with_insights(
                    few_shots=few_shots,
                    memory_few_shots=successful_shots,
                    insights=roles_rules.get(ground_truth.profile, raw_rules),
                    task_description=self.meta_memory.summarize()
                )
                if retriever_action != '':
                    user_prompt = f"{user_prompt}\n\nRetriever suggestion: {retriever_action}"
                tries = 0
                while tries < 3:
                    try:
                        action: str = ground_truth.response(user_prompt, self.reasoning_config)
                        if action == '':
                            continue
                        action = env.process_action(action)
                        break
                    except Exception as e:
                        print(f'Error during execution of ground truth agent: {e}')
                    tries += 1
                self.notify_observers(f"[Turn {i+1}] GroundTruth action: {_short(action, 120)}")

                # 更新ground_truth的私有memory：记录新动作
                if ground_truth.private_memory:
                    ground_truth.private_memory.add_reasoning_step(
                        step_id=f"intervene_action_{i}",
                        premise=f"Solver stuck with {action_history[-1]}, need alternative approach",
                        conclusion=f"Proposed fresh action: {action}",
                        reasoning_type="abduction",
                        confidence=0.9
                    )
                    ground_truth.private_memory.update_task_status(
                        task_id=f"intervene_{i}",
                        status="completed",
                        next_action=None
                    )

                name: str = ground_truth.name
                system_instruction = ground_truth.system_instruction
            
            agent_message: AgentMessage = AgentMessage(
                agent_name=name,
                system_instruction=system_instruction,
                user_instruction=user_prompt,
                message=action,
            )
            self.meta_memory.add_agent_node(agent_message, upstream_agent_ids=[])

            observation, reward, done = env.step(action)
            action_history.append(action)

            self.notify_observers(
                f"[Turn {i+1}] Actor={name} action={_short(action, 120)} | reward={reward} | done={done}"
            )
            step_message: str = f'Act {i + 1}: {action}\nObs {i + 1}: {observation}'
            self.notify_observers(step_message)

            self.meta_memory.move_memory_state(action, observation, reward=reward)

            # ========== 更新所有相关agent的私有memory（基于环境反馈）==========
            # 更新执行agent（solver或ground_truth）的memory
            if name == solver.name and solver.private_memory:
                solver.private_memory.add_key_observation(f"Act: {action[:40]} | Obs: {observation[:60]}")
                solver.private_memory.add_evidence(
                    evidence_id=f"obs_{i}",
                    content=observation[:200],
                    source="environment",
                    relevance=0.9
                )
                # 根据reward更新假设置信度
                if reward > 0:
                    solver.private_memory.update_state_summary(f"Progress at turn {i+1}, reward: {reward:.2f}")
                    self.notify_observers(f"[Private Memory] Solver: Positive reward {reward:.2f}, updated state")

                # 如果使用了retriever建议，更新协作成功率
                if retriever_action != '':
                    solver.private_memory.record_peer_interaction(
                        peer_name=retriever.name,
                        success=(reward > 0)
                    )
                    self.notify_observers(f"[Private Memory] Solver: Updated Retriever collaboration success")

            elif name == ground_truth.name and ground_truth.private_memory:
                ground_truth.private_memory.add_key_observation(f"Intervention result: {observation[:60]}")
                ground_truth.private_memory.add_evidence(
                    evidence_id=f"intervention_result_{i}",
                    content=f"After breaking loop, got: {observation[:150]}",
                    source="environment",
                    relevance=1.0
                )
                self.notify_observers(f"[Private Memory] GroundTruth: Recorded intervention result")

            # ========== 更新共享记忆：记录环境反馈 ==========
            # ========== Curator: structured memory write ==========
            curated = {}
            if getattr(self, "_memory_curator", None):
                curated = self._memory_curator.curate(task_main, action, observation)

            curated_answer = curated.get("answer") if isinstance(curated, dict) else None
            curated_next = curated.get("next_queries", []) if isinstance(curated, dict) else []
            curated_facts = curated.get("facts", []) if isinstance(curated, dict) else []
            curated_relations = curated.get("relations", []) if isinstance(curated, dict) else []

            # Log curator JSON to prompt logs for debugging
            if isinstance(curated, dict):
                has_signal = bool(
                    curated.get("answer")
                    or curated.get("facts")
                    or curated.get("relations")
                    or curated.get("next_queries")
                    or curated.get("entities")
                )
                if has_signal:
                    try:
                        curated_json = json.dumps(curated, ensure_ascii=False)
                    except Exception:
                        curated_json = str(curated)
                    self.notify_observers(f"[PromptLog][Curator][Turn {i+1}] {curated_json}")

            # Private Memory -> LocalPlan / EvidenceTrace (solver only)
            if solver.private_memory:
                solver.private_memory.add_task(
                    task_id=f"hop_{i}",
                    description=f"{action}",
                    status="completed" if curated_answer else "in_progress",
                    priority=2
                )
                for j, q in enumerate(curated_next[:2]):
                    if not q:
                        continue
                    q_text = q if q.startswith("Search[") else f"Search[{q}]"
                    solver.private_memory.add_task(
                        task_id=f"next_{i}_{j}",
                        description=q_text,
                        status="pending",
                        priority=1,
                        dependencies=[f"hop_{i}"]
                    )

                for k in curated_facts:
                    key = k.get("key") if isinstance(k, dict) else None
                    val = k.get("value") if isinstance(k, dict) else None
                    if not key or val is None:
                        continue
                    solver.private_memory.add_evidence(
                        evidence_id=f"fact_{i}_{key}",
                        content=f"{key} = {val}",
                        source="memory_curator"
                    )

                for r in curated_relations:
                    if not isinstance(r, dict):
                        continue
                    head = r.get("head")
                    rel = r.get("relation")
                    tail = r.get("tail")
                    if not head or not rel or not tail:
                        continue
                    solver.private_memory.add_reasoning_step(
                        step_id=f"rel_{i}",
                        premise=f"{head} --{rel}--> {tail}",
                        conclusion=f"Bridge: {tail}",
                        reasoning_type="deduction"
                    )

                if curated_answer:
                    solver.private_memory.add_hypothesis(
                        hypo_id=f"ans_{i}",
                        content=f"Candidate answer: {curated_answer}"
                    )

            # ========== Shared Memory update ==========
            hop_query = action
            match = re.match(r"^\w+\[(.+)\]$", action)
            if match:
                hop_query = match.group(1).strip()

            shared_mem.add_hop(
                query=hop_query,
                action=action,
                observation=observation,
                extracted_answer=curated_answer,
                agent=name,
                success=(reward >= 0 and "Invalid" not in observation)
            )

            for k in curated_facts:
                key = k.get("key") if isinstance(k, dict) else None
                val = k.get("value") if isinstance(k, dict) else None
                if key and val is not None:
                    shared_mem.add_fact(key, val)

            if shared_mem.task_status and curated_next:
                est = shared_mem.task_status.current_hop + len(curated_next)
                shared_mem.task_status.total_hops_estimated = max(
                    shared_mem.task_status.total_hops_estimated, est
                )

            if retriever_action != '':
                shared_mem.record_collaboration(
                    agent1=retriever.name,
                    agent2=solver.name,
                    interaction_type="suggestion_execution",
                    outcome="success" if reward > 0 else "failed"
                )

            # 打印共享记忆状态（调试）
            if i % 5 == 0 or done:  # 每5轮或任务结束时打印
                stats = shared_mem.get_statistics()
                self.notify_observers(f"[Shared Memory] Stats: {stats['total_hops']} hops, {stats['successful_hops']} success, {stats['unique_searches']} searches")

            if done:
                break

        # ========== Final feedback and memory update ==========
        final_reward, final_done, final_feedback = self.env.feedback()
        self.notify_observers(final_feedback)
        self.meta_memory.save_task_context(label=final_done, feedback=final_feedback)
        self.meta_memory.backward(final_done)

        # ========== 任务结束后的Memory更新 ==========

        # 1. 更新私有记忆
        if final_done:
            if solver.private_memory:
                solver.private_memory.update_task_status("main_task", "completed")
                self.notify_observers(f"[Private Memory] Solver: Task completed successfully")
        else:
            if solver.private_memory:
                solver.private_memory.add_challenge(f"Task failed: {final_feedback[:100]}")
                self.notify_observers(f"[Private Memory] Solver: Recorded failure challenge")

        # 2. 更新共享记忆
        if final_done:
            shared_mem.mark_completed(final_answer=final_feedback)
        else:
            shared_mem.mark_failed()

        # 3. 打印最终Memory统计（调试）
        self.notify_observers("\n" + "="*60)
        self.notify_observers("[Memory Summary - Task Completed]")

        # 共享记忆统计
        shared_stats = shared_mem.get_statistics()
        self.notify_observers(f"[Shared Memory] Final Stats:")
        self.notify_observers(f"  - Total hops: {shared_stats['total_hops']}")
        self.notify_observers(f"  - Successful hops: {shared_stats['successful_hops']}")
        self.notify_observers(f"  - Failed hops: {shared_stats['failed_hops']}")
        self.notify_observers(f"  - Unique searches: {shared_stats['unique_searches']}")
        self.notify_observers(f"  - Agent interactions: {shared_stats['agent_interactions']}")

        # 私有记忆统计
        if solver.private_memory:
            solver_summary = solver.private_memory.get_compact_summary()
            self.notify_observers(f"[Private Memory] Solver: {solver_summary}")
            self.notify_observers(f"  - Total turns: {solver.private_memory.turn_counter}")
            self.notify_observers(f"  - Hypotheses count: {len(solver.private_memory.hypotheses_store.hypotheses)}")
            self.notify_observers(f"  - Evidence count: {len(solver.private_memory.evidence_trace.evidences)}")
            self.notify_observers(f"  - Peer profiles: {len(solver.private_memory.peer_profiles)}")

        if retriever.private_memory:
            self.notify_observers(f"[Private Memory] Retriever: {retriever.private_memory.turn_counter} turns, {len(retriever.private_memory.evidence_trace.evidences)} evidence")

        self.notify_observers("="*60 + "\n")

        return final_reward, final_done   
    
    def _solver_stuck(self, current_action: str, action_history: list[str]) -> bool:
        """
        Determines whether the agent is stuck by repeating the same action.

        If the current action is identical to the last two actions in the history,
        the agent is considered to be stuck in a loop.

        Args:
            current_action (str): The action currently being executed by the agent.
            action_history (list[str]): A chronological list of previously taken actions.

        Returns:
            bool: True if the last two actions are the same as the current action; False otherwise.
        """
        return len(action_history) >= 2 and current_action == action_history[-1] and current_action == action_history[-2]

    def _initialize_agent_contracts(self, env: Env):
        """为所有agent初始化角色契约（根据环境类型定制）"""
        solver = self.get_agent(self.solver_name)
        retriever = self.get_agent(self.retriever_name)
        ground_truth = self.get_agent(self.ground_truth_name)

        # 获取环境允许的动作
        env_actions = getattr(env, 'action_types', ['Think', 'Search', 'Lookup', 'Finish'])

        # Solver角色契约
        if solver.private_memory:
            solver.private_memory.set_role_contract(
                responsibilities=[
                    "分析问题并制定解决方案",
                    "综合retriever的建议做出决策",
                    "生成最终答案或动作"
                ],
                boundaries={
                    "can_do": ["推理", "选择动作", "使用retriever建议"],
                    "cannot_do": ["直接访问ground truth", "跳过必要的检索步骤"]
                },
                priorities=["accuracy > speed", "evidence-based > speculation"],
                allowed_tools=env_actions,
                output_format="Action[argument]"
            )

        # Retriever角色契约
        if retriever.private_memory:
            retriever.private_memory.set_role_contract(
                responsibilities=[
                    "提出有效的检索策略",
                    "建议Search或Lookup动作",
                    "避免重复检索"
                ],
                boundaries={
                    "can_do": ["建议检索动作", "分析检索需求"],
                    "cannot_do": ["直接回答问题", "输出Finish动作"]
                },
                priorities=["recall > precision", "diverse queries > repeated queries"],
                allowed_tools=["Search", "Lookup"],
                output_format="Search[entity] or Lookup[term]"
            )

        # Ground Truth角色契约
        if ground_truth.private_memory:
            ground_truth.private_memory.set_role_contract(
                responsibilities=[
                    "在solver陷入循环时提供正确引导",
                    "避免重复solver的错误思路",
                    "提供突破性建议"
                ],
                boundaries={
                    "can_do": ["分析solver失败原因", "提供替代方案"],
                    "cannot_do": ["重复solver的错误", "使用相同的推理路径"]
                },
                priorities=["novelty > consistency", "break loops > follow patterns"],
                allowed_tools=env_actions,
                output_format="Action[argument] with fresh perspective"
            )

    def _project_insights(self, insights: list[str]) -> dict[str, list[str]]:
        """
        Process insights to generate a dictionary matching roles to insights, based on whether a projector is used.

        Args:
            insights (list[str]): A list of insight strings.

        Returns:
            dict[str, list[str]]: A dictionary with roles as keys and lists of insights as values.
        """

        roles_rules: dict[str, list[str]] = {}
        roles = set([agent.profile for agent in self.agents_team.values()])

        if not self._use_projector or not isinstance(self.meta_memory, GMemory):
            for role in roles:
                roles_rules[role] = insights
        else:
            for role in roles:
                roles_rules[role] = self.meta_memory.project_insights(insights, role)
        
        # Limit the number of insights per role to self._insights_topk
        for role, insights in roles_rules.items():
            roles_rules[role] = insights[:self._insights_topk]
        return roles_rules
        
            
