"""
Agent Private Memory Module
单次任务内的agent私有记忆系统，用于优化reasoning过程中的上下文管理
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


# ========== 数据结构定义 ==========

@dataclass
class RoleContract:
    """角色身份与工作方式"""
    role_name: str  # 角色名称（solver/retriever/verifier等）
    responsibilities: List[str] = field(default_factory=list)  # 职责清单
    boundaries: Dict[str, List[str]] = field(default_factory=dict)  # {"can_do": [...], "cannot_do": [...]}
    priorities: List[str] = field(default_factory=list)  # 优先级策略，如["accuracy > speed", "safety > cost"]
    allowed_tools: List[str] = field(default_factory=list)  # 允许使用的工具/动作
    output_format: Optional[str] = None  # 输出格式偏好


@dataclass
class TaskItem:
    """单个子任务"""
    task_id: str
    description: str
    status: str  # "pending" | "in_progress" | "completed" | "blocked"
    priority: int = 0  # 优先级，数字越大越优先
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他任务ID或agent名
    next_action: Optional[str] = None  # 下一步具体动作
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LocalPlan:
    """本角色的任务分解与待办"""
    tasks: Dict[str, TaskItem] = field(default_factory=dict)  # {task_id: TaskItem}
    current_focus: Optional[str] = None  # 当前聚焦的任务ID

    def add_task(self, task: TaskItem):
        self.tasks[task.task_id] = task

    def get_pending_tasks(self) -> List[TaskItem]:
        """获取待处理任务，按优先级排序"""
        return sorted(
            [t for t in self.tasks.values() if t.status == "pending"],
            key=lambda x: x.priority,
            reverse=True
        )

    def update_status(self, task_id: str, status: str):
        if task_id in self.tasks:
            self.tasks[task_id].status = status


@dataclass
class Hypothesis:
    """单个假设/备选方案"""
    hypo_id: str
    content: str  # 假设内容
    confidence: float = 0.5  # 置信度 [0, 1]
    verified: bool = False  # 是否已验证
    pros: List[str] = field(default_factory=list)  # 优点
    cons: List[str] = field(default_factory=list)  # 缺点
    risks: List[str] = field(default_factory=list)  # 风险点


@dataclass
class HypothesesStore:
    """假设、草案与备选方案"""
    hypotheses: Dict[str, Hypothesis] = field(default_factory=dict)
    active_hypothesis: Optional[str] = None  # 当前采用的假设ID

    def add_hypothesis(self, hypo: Hypothesis):
        self.hypotheses[hypo.hypo_id] = hypo

    def get_top_hypotheses(self, k: int = 3) -> List[Hypothesis]:
        """获取置信度最高的k个假设"""
        return sorted(
            self.hypotheses.values(),
            key=lambda x: x.confidence,
            reverse=True
        )[:k]


@dataclass
class EvidencePiece:
    """单条证据"""
    evidence_id: str
    content: str  # 证据内容
    source: str  # 来源（观察/推理/外部）
    relevance: float = 1.0  # 相关度 [0, 1]
    supports: Optional[str] = None  # 支持哪个假设ID
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: str
    premise: str  # 前提
    conclusion: str  # 结论
    reasoning_type: str  # "deduction" | "induction" | "abduction"
    confidence: float = 1.0


@dataclass
class EvidenceTrace:
    """专业知识片段与推理痕迹"""
    evidences: Dict[str, EvidencePiece] = field(default_factory=dict)
    reasoning_chain: List[ReasoningStep] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)  # 对共享事实的质疑点

    def add_evidence(self, evidence: EvidencePiece):
        self.evidences[evidence.evidence_id] = evidence

    def add_reasoning_step(self, step: ReasoningStep):
        self.reasoning_chain.append(step)

    def get_recent_reasoning(self, k: int = 5) -> List[ReasoningStep]:
        """获取最近k步推理"""
        return self.reasoning_chain[-k:]


@dataclass
class PeerProfile:
    """对其他agent的画像（轻量）"""
    peer_name: str
    expertise: List[str] = field(default_factory=list)  # 擅长领域
    reliable_output_types: List[str] = field(default_factory=list)  # 可靠输出类型
    current_status: Optional[str] = None  # 当前状态描述
    interaction_count: int = 0  # 交互次数
    success_rate: float = 0.0  # 成功率（可选）


@dataclass
class InteractionRecord:
    """交互记录"""
    turn: int
    peer_name: str
    message_type: str  # "request" | "response" | "suggestion"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class WorkingContext:
    """交互上下文缓存（短期工作记忆）"""
    recent_interactions: List[InteractionRecord] = field(default_factory=list)
    tool_results_cache: Dict[str, Any] = field(default_factory=dict)  # {tool_name: result}
    key_observations: List[str] = field(default_factory=list)  # 关键观察
    current_state_summary: Optional[str] = None  # 当前状态摘要
    max_interactions: int = 10  # 最多保留交互记录数

    def add_interaction(self, record: InteractionRecord):
        self.recent_interactions.append(record)
        # 保持队列长度
        if len(self.recent_interactions) > self.max_interactions:
            self.recent_interactions = self.recent_interactions[-self.max_interactions:]

    def get_recent_interactions(self, k: int = 5) -> List[InteractionRecord]:
        """获取最近k轮交互"""
        return self.recent_interactions[-k:]


# ========== 主Memory类 ==========

class AgentPrivateMemory:
    """
    Agent私有记忆系统
    生命周期：单次任务内（从reset到任务结束）
    """

    def __init__(
        self,
        agent_name: str,
        role: str,
        max_interaction_history: int = 10,
        max_evidence: int = 50,
        max_hypotheses: int = 10
    ):
        self.agent_name = agent_name

        # 六大模块
        self.role_contract = RoleContract(role_name=role)
        self.local_plan = LocalPlan()
        self.hypotheses_store = HypothesesStore()
        self.evidence_trace = EvidenceTrace()
        self.peer_profiles: Dict[str, PeerProfile] = {}  # {peer_name: PeerProfile}
        self.working_context = WorkingContext(max_interactions=max_interaction_history)

        # 容量限制
        self.max_evidence = max_evidence
        self.max_hypotheses = max_hypotheses

        # 任务元信息
        self.task_start_time = datetime.now().isoformat()
        self.turn_counter = 0

    # ========== 1. Role Contract 操作 ==========

    def set_role_contract(
        self,
        responsibilities: List[str] = None,
        boundaries: Dict[str, List[str]] = None,
        priorities: List[str] = None,
        allowed_tools: List[str] = None,
        output_format: str = None
    ):
        """设置角色契约"""
        if responsibilities:
            self.role_contract.responsibilities = responsibilities
        if boundaries:
            self.role_contract.boundaries = boundaries
        if priorities:
            self.role_contract.priorities = priorities
        if allowed_tools:
            self.role_contract.allowed_tools = allowed_tools
        if output_format:
            self.role_contract.output_format = output_format

    # ========== 2. Local Plan 操作 ==========

    def add_task(
        self,
        task_id: str,
        description: str,
        status: str = "pending",
        priority: int = 0,
        dependencies: List[str] = None,
        next_action: str = None
    ):
        """添加子任务"""
        task = TaskItem(
            task_id=task_id,
            description=description,
            status=status,
            priority=priority,
            dependencies=dependencies or [],
            next_action=next_action
        )
        self.local_plan.add_task(task)

    def update_task_status(self, task_id: str, status: str, next_action: str = None):
        """更新任务状态"""
        self.local_plan.update_status(task_id, status)
        if next_action and task_id in self.local_plan.tasks:
            self.local_plan.tasks[task_id].next_action = next_action

    # ========== 3. Hypotheses Store 操作 ==========

    def add_hypothesis(
        self,
        hypo_id: str,
        content: str,
        confidence: float = 0.5,
        pros: List[str] = None,
        cons: List[str] = None,
        risks: List[str] = None
    ):
        """添加假设"""
        hypo = Hypothesis(
            hypo_id=hypo_id,
            content=content,
            confidence=confidence,
            pros=pros or [],
            cons=cons or [],
            risks=risks or []
        )
        self.hypotheses_store.add_hypothesis(hypo)

        # 容量控制：保留置信度最高的
        if len(self.hypotheses_store.hypotheses) > self.max_hypotheses:
            top_hypos = self.hypotheses_store.get_top_hypotheses(self.max_hypotheses)
            self.hypotheses_store.hypotheses = {h.hypo_id: h for h in top_hypos}

    def verify_hypothesis(self, hypo_id: str, verified: bool):
        """验证假设"""
        if hypo_id in self.hypotheses_store.hypotheses:
            self.hypotheses_store.hypotheses[hypo_id].verified = verified

    # ========== 4. Evidence Trace 操作 ==========

    def add_evidence(
        self,
        evidence_id: str,
        content: str,
        source: str,
        relevance: float = 1.0,
        supports: str = None
    ):
        """添加证据"""
        evidence = EvidencePiece(
            evidence_id=evidence_id,
            content=content,
            source=source,
            relevance=relevance,
            supports=supports
        )
        self.evidence_trace.add_evidence(evidence)

        # 容量控制：保留相关度最高的
        if len(self.evidence_trace.evidences) > self.max_evidence:
            sorted_evidences = sorted(
                self.evidence_trace.evidences.values(),
                key=lambda x: x.relevance,
                reverse=True
            )[:self.max_evidence]
            self.evidence_trace.evidences = {e.evidence_id: e for e in sorted_evidences}

    def add_reasoning_step(
        self,
        step_id: str,
        premise: str,
        conclusion: str,
        reasoning_type: str = "deduction",
        confidence: float = 1.0
    ):
        """添加推理步骤"""
        step = ReasoningStep(
            step_id=step_id,
            premise=premise,
            conclusion=conclusion,
            reasoning_type=reasoning_type,
            confidence=confidence
        )
        self.evidence_trace.add_reasoning_step(step)

    def add_challenge(self, challenge: str):
        """添加对共享事实的质疑"""
        self.evidence_trace.challenges.append(challenge)

    # ========== 5. Peer Profile 操作 ==========

    def update_peer_profile(
        self,
        peer_name: str,
        expertise: List[str] = None,
        reliable_output_types: List[str] = None,
        current_status: str = None
    ):
        """更新对其他agent的画像"""
        if peer_name not in self.peer_profiles:
            self.peer_profiles[peer_name] = PeerProfile(peer_name=peer_name)

        profile = self.peer_profiles[peer_name]
        if expertise:
            profile.expertise = expertise
        if reliable_output_types:
            profile.reliable_output_types = reliable_output_types
        if current_status:
            profile.current_status = current_status

    def record_peer_interaction(self, peer_name: str, success: bool):
        """记录与peer的交互结果，更新成功率"""
        if peer_name not in self.peer_profiles:
            self.peer_profiles[peer_name] = PeerProfile(peer_name=peer_name)

        profile = self.peer_profiles[peer_name]
        profile.interaction_count += 1
        # 指数移动平均更新成功率
        alpha = 0.3
        profile.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * profile.success_rate

    # ========== 6. Working Context 操作 ==========

    def add_interaction(
        self,
        peer_name: str,
        message_type: str,
        content: str
    ):
        """添加交互记录"""
        self.turn_counter += 1
        record = InteractionRecord(
            turn=self.turn_counter,
            peer_name=peer_name,
            message_type=message_type,
            content=content
        )
        self.working_context.add_interaction(record)

    def cache_tool_result(self, tool_name: str, result: Any):
        """缓存工具结果"""
        self.working_context.tool_results_cache[tool_name] = result

    def add_key_observation(self, observation: str):
        """添加关键观察"""
        self.working_context.key_observations.append(observation)
        # 保持最近20条
        if len(self.working_context.key_observations) > 20:
            self.working_context.key_observations = self.working_context.key_observations[-20:]

    def update_state_summary(self, summary: str):
        """更新当前状态摘要"""
        self.working_context.current_state_summary = summary

    # ========== 记忆检索与格式化（核心功能）==========

    def format_for_prompt(
        self,
        include_role_contract: bool = True,
        include_plan: bool = True,
        include_hypotheses: bool = True,
        include_evidence: bool = True,
        include_peers: bool = False,
        include_context: bool = True,
        max_recent_interactions: int = 5,
        max_recent_reasoning: int = 3,
        max_top_hypotheses: int = 3,
        compact: bool = False
    ) -> str:
        """
        格式化记忆为prompt字符串，用于添加到agent的上下文中

        Args:
            include_*: 控制包含哪些模块
            max_*: 控制各模块的条目数
            compact: 是否使用紧凑格式（节省token）

        Returns:
            格式化的记忆字符串
        """
        sections = []
        separator = "\n" if compact else "\n\n"

        # 1. Role Contract
        if include_role_contract and self.role_contract.responsibilities:
            role_text = f"【Role】{self.role_contract.role_name}"
            if self.role_contract.priorities:
                role_text += f" | Priorities: {', '.join(self.role_contract.priorities[:2])}"
            if self.role_contract.allowed_tools:
                role_text += f" | Tools: {', '.join(self.role_contract.allowed_tools[:5])}"
            sections.append(role_text)

        # 2. Local Plan
        if include_plan:
            pending = self.local_plan.get_pending_tasks()
            in_progress = [t for t in self.local_plan.tasks.values() if t.status == "in_progress"]

            if pending or in_progress:
                plan_lines = ["【Plan】"]
                if in_progress:
                    for task in in_progress[:2]:
                        plan_lines.append(f"→ {task.description} [{task.status}]")
                if pending:
                    for task in pending[:3]:
                        next_act = f" → {task.next_action}" if task.next_action else ""
                        plan_lines.append(f"• {task.description}{next_act}")
                sections.append("\n".join(plan_lines))

        # 3. Hypotheses
        if include_hypotheses:
            top_hypos = self.hypotheses_store.get_top_hypotheses(max_top_hypotheses)
            if top_hypos:
                hypo_lines = ["【Hypotheses】"]
                for h in top_hypos:
                    status = "✓" if h.verified else f"~{h.confidence:.1f}"
                    risks_txt = f" ⚠ {h.risks[0]}" if h.risks else ""
                    hypo_lines.append(f"{status} {h.content}{risks_txt}")
                sections.append("\n".join(hypo_lines))

        # 4. Evidence & Reasoning
        if include_evidence:
            recent_reasoning = self.evidence_trace.get_recent_reasoning(max_recent_reasoning)
            if recent_reasoning:
                ev_lines = ["【Reasoning】"]
                for step in recent_reasoning:
                    ev_lines.append(f"∵ {step.premise} ∴ {step.conclusion}")
                sections.append("\n".join(ev_lines))

            # 关键证据
            top_evidences = sorted(
                self.evidence_trace.evidences.values(),
                key=lambda x: x.relevance,
                reverse=True
            )[:3]
            if top_evidences:
                sections.append(
                    "【Evidence】" + "; ".join([f"{e.content} ({e.source})" for e in top_evidences])
                )

        # 5. Peer Profiles（可选，默认关闭以节省token）
        if include_peers and self.peer_profiles:
            peer_lines = ["【Peers】"]
            for peer in list(self.peer_profiles.values())[:3]:
                status = f" ({peer.current_status})" if peer.current_status else ""
                peer_lines.append(f"• {peer.peer_name}: {', '.join(peer.expertise[:2])}{status}")
            sections.append("\n".join(peer_lines))

        # 6. Working Context
        if include_context:
            context_parts = []

            # 当前状态
            if self.working_context.current_state_summary:
                context_parts.append(f"State: {self.working_context.current_state_summary}")

            # 关键观察（最近3条）
            recent_obs = self.working_context.key_observations[-3:]
            if recent_obs:
                context_parts.append(f"Recent: {' | '.join(recent_obs)}")

            # 最近交互
            recent_inter = self.working_context.get_recent_interactions(max_recent_interactions)
            if recent_inter:
                inter_txt = "; ".join([
                    f"{r.peer_name}→{r.content[:30]}" for r in recent_inter
                ])
                context_parts.append(f"History: {inter_txt}")

            if context_parts:
                sections.append("【Context】" + " | ".join(context_parts))

        return separator.join(sections)

    def get_compact_summary(self) -> str:
        """获取超紧凑摘要（用于token受限场景）"""
        parts = []

        # 当前任务
        in_progress = [t for t in self.local_plan.tasks.values() if t.status == "in_progress"]
        if in_progress:
            parts.append(f"Task: {in_progress[0].description}")

        # 最高置信度假设
        top_hypo = self.hypotheses_store.get_top_hypotheses(1)
        if top_hypo:
            parts.append(f"Hypothesis: {top_hypo[0].content}")

        # 最近推理
        recent = self.evidence_trace.get_recent_reasoning(1)
        if recent:
            parts.append(f"Reasoning: {recent[0].conclusion}")

        return " | ".join(parts)

    # ========== 记忆持久化 ==========

    def to_dict(self) -> Dict[str, Any]:
        """转为字典（用于序列化）"""
        return {
            "agent_name": self.agent_name,
            "task_start_time": self.task_start_time,
            "turn_counter": self.turn_counter,
            "role_contract": asdict(self.role_contract),
            "local_plan": {
                "tasks": {k: asdict(v) for k, v in self.local_plan.tasks.items()},
                "current_focus": self.local_plan.current_focus
            },
            "hypotheses_store": {
                "hypotheses": {k: asdict(v) for k, v in self.hypotheses_store.hypotheses.items()},
                "active_hypothesis": self.hypotheses_store.active_hypothesis
            },
            "evidence_trace": {
                "evidences": {k: asdict(v) for k, v in self.evidence_trace.evidences.items()},
                "reasoning_chain": [asdict(s) for s in self.evidence_trace.reasoning_chain],
                "challenges": self.evidence_trace.challenges
            },
            "peer_profiles": {k: asdict(v) for k, v in self.peer_profiles.items()},
            "working_context": {
                "recent_interactions": [asdict(r) for r in self.working_context.recent_interactions],
                "tool_results_cache": self.working_context.tool_results_cache,
                "key_observations": self.working_context.key_observations,
                "current_state_summary": self.working_context.current_state_summary
            }
        }

    def to_json(self, file_path: str = None) -> str:
        """导出为JSON"""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        return json_str

    def reset(self):
        """重置记忆（用于新任务开始）"""
        self.local_plan = LocalPlan()
        self.hypotheses_store = HypothesesStore()
        self.evidence_trace = EvidenceTrace()
        self.peer_profiles = {}
        self.working_context = WorkingContext(max_interactions=self.working_context.max_interactions)
        self.turn_counter = 0
        self.task_start_time = datetime.now().isoformat()
