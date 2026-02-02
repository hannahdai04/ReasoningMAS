"""
Shared Memory - 轻量级多agent共享记忆
用于多跳推理任务中的状态同步和避免重复劳动
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from mas.memory.utils import cosine_similarity


@dataclass
class ReasoningHop:
    """单次推理跳"""
    hop_id: int
    query: str  # 本跳要回答的问题
    action: str  # 执行的动作（如 Search[...]）
    observation: str  # 环境返回的结果
    extracted_answer: Optional[str] = None  # 从observation中提取的答案
    agent: str = "solver"  # 哪个agent执行的
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True  # 本跳是否成功


@dataclass
class TaskStatus:
    """任务整体状态"""
    task_description: str
    current_hop: int = 0
    total_hops_estimated: int = 2  # 估计需要几跳（可动态调整）
    status: str = "in_progress"  # "in_progress" | "completed" | "failed"
    final_answer: Optional[str] = None


class SharedMemory:
    """
    共享记忆（单例模式）
    所有agent可读可写，用于同步任务状态

    核心功能：
    1. 推理链追踪（避免丢失中间结果）
    2. 检索去重（避免重复Search）
    3. 任务状态可视化
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 1. 推理链（核心数据结构）
        self.reasoning_chain: List[ReasoningHop] = []

        # 2. 检索历史（去重用）
        self.search_history: Dict[str, str] = {}  # {query: result}

        # 3. 任务状态
        self.task_status: Optional[TaskStatus] = None

        # 4. 已提取的中间实体/事实
        self.accumulated_facts: Dict[str, Any] = {}

        # 5. Agent协作记录
        self.agent_interactions: List[Dict] = []

        self._embedding_func = None
        self._embedding_cache: Dict[str, list] = {}

        self._initialized = True

    # ========== 初始化与重置 ==========

    def set_embedding_func(self, embedding_func: Any):
        """Inject an embedding function for vector retrieval."""
        self._embedding_func = embedding_func

    def reset(self, task_description: str = ""):
        """重置共享记忆（新任务开始时调用）"""
        self.reasoning_chain.clear()
        self.search_history.clear()
        self.accumulated_facts.clear()
        self.agent_interactions.clear()
        self._embedding_cache = {}
        self.task_status = TaskStatus(task_description=task_description)

    # ========== 推理链管理 ==========

    def add_hop(
        self,
        query: str,
        action: str,
        observation: str,
        extracted_answer: str = None,
        agent: str = "solver",
        success: bool = True
    ) -> int:
        """
        添加新的推理跳
        返回: hop_id
        """
        hop_id = len(self.reasoning_chain) + 1

        hop = ReasoningHop(
            hop_id=hop_id,
            query=query,
            action=action,
            observation=observation,
            extracted_answer=extracted_answer,
            agent=agent,
            success=success
        )

        self.reasoning_chain.append(hop)

        # 更新任务状态
        if self.task_status:
            self.task_status.current_hop = hop_id

        return hop_id

    def get_last_hop(self) -> Optional[ReasoningHop]:
        """获取最后一跳"""
        return self.reasoning_chain[-1] if self.reasoning_chain else None

    def get_hop(self, hop_id: int) -> Optional[ReasoningHop]:
        """获取指定跳"""
        for hop in self.reasoning_chain:
            if hop.hop_id == hop_id:
                return hop
        return None

    # ========== 检索去重 ==========

    def has_searched(self, query: str) -> bool:
        """检查是否已检索过该query"""
        return query in self.search_history

    def add_search_result(self, query: str, result: str):
        """记录检索结果"""
        self.search_history[query] = result

    def get_search_result(self, query: str) -> Optional[str]:
        """获取之前的检索结果"""
        return self.search_history.get(query)

    # ========== 中间事实累积 ==========

    def add_fact(self, key: str, value: Any):
        """添加中间事实（如：director="James Cameron"）"""
        self.accumulated_facts[key] = value

    def get_fact(self, key: str) -> Optional[Any]:
        """获取已知事实"""
        return self.accumulated_facts.get(key)

    # ========== Agent交互记录 ==========

    def record_interaction(self, from_agent: str, to_agent: str, message: str):
        """记录agent间的交互"""
        self.agent_interactions.append({
            "from": from_agent,
            "to": to_agent,
            "message": message[:100],  # 截断过长消息
            "timestamp": datetime.now().isoformat()
        })

    # ========== 任务状态管理 ==========

    def mark_completed(self, final_answer: str):
        """标记任务完成"""
        if self.task_status:
            self.task_status.status = "completed"
            self.task_status.final_answer = final_answer

    def mark_failed(self):
        """标记任务失败"""
        if self.task_status:
            self.task_status.status = "failed"

    # ========== 新增：AutoGen集成所需方法 ==========

    def retrieve_relevant_insights(self, query: str, topk: int = 3) -> List[str]:
        """
        检索相关的insights（从推理链中提取经验）

        Args:
            query: 当前任务查询
            topk: 返回top-k条insights

        Returns:
            insights列表
        """
        insights = []

        # 从成功的推理跳中提取insights
        for hop in self.reasoning_chain:
            if hop.success and hop.extracted_answer:
                insight = f"When dealing with '{hop.query[:50]}', using {hop.action} yielded: {hop.extracted_answer}"
                insights.append(insight)

        # 返回最近的topk条
        return insights[-topk:] if insights else []

    def get_collaboration_history(self) -> List[Dict]:
        """
        获取协作历史记录

        Returns:
            协作交互列表
        """
        return self.agent_interactions[-10:]  # 返回最近10条

    def add_agent_action(self, agent_name: str, action_type: str, content: str, turn: int):
        """
        记录agent的动作到推理链

        Args:
            agent_name: agent名称
            action_type: 动作类型（suggestion/decision/intervention）
            content: 动作内容
            turn: 当前轮次
        """
        # 简化记录为interaction
        self.record_interaction(
            from_agent=agent_name,
            to_agent="system",
            message=f"[{action_type}] {content}"
        )

    def record_collaboration(self, agent1: str, agent2: str, interaction_type: str, outcome: str):
        """
        记录agent间的协作

        Args:
            agent1: 发起agent
            agent2: 接收agent
            interaction_type: 交互类型
            outcome: 结果（pending/success/failed）
        """
        self.record_interaction(
            from_agent=agent1,
            to_agent=agent2,
            message=f"[{interaction_type}] outcome={outcome}"
        )

    def get_global_context(self) -> str:
        """
        获取全局上下文摘要

        Returns:
            上下文字符串
        """
        if self.task_status:
            return f"{self.task_status.task_description} (Hop {self.task_status.current_hop}/{self.task_status.total_hops_estimated})"
        return "No active task"

    # ========== 检索功能（关键！）==========

    def retrieve_relevant(
        self,
        current_query: str,
        current_state: str = "",
        top_k: int = 3,
        use_vector: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        基于当前查询和状态检索相关的共享记忆

        Args:
            current_query: 当前要解决的问题/任务
            current_state: 当前任务状态描述
            top_k: 返回最相关的top-k条记忆

        Returns:
            检索结果字典，包含相关的hop、facts、searches
        """
        if use_vector is None:
            use_vector = self._embedding_func is not None

        if use_vector and self._embedding_func is not None:
            return self._retrieve_relevant_vector(current_query=current_query, top_k=top_k)

        results = {
            "relevant_hops": [],
            "relevant_facts": {},
            "relevant_searches": {},
            "task_context": None
        }

        # 1. 任务上下文（总是返回）
        if self.task_status:
            results["task_context"] = {
                "description": self.task_status.task_description,
                "current_hop": self.task_status.current_hop,
                "status": self.task_status.status
            }

        # 2. 检索相关的推理跳（基于关键词匹配 - 生产环境可用向量检索）
        if self.reasoning_chain:
            # 提取查询关键词
            query_keywords = set(current_query.lower().split())

            # 计算每个hop的相关性分数
            hop_scores = []
            for hop in self.reasoning_chain:
                # 简单的关键词重叠计数
                hop_text = f"{hop.query} {hop.action} {hop.observation}".lower()
                hop_keywords = set(hop_text.split())
                overlap = len(query_keywords & hop_keywords)
                hop_scores.append((hop, overlap))

            # 按分数排序，取top-k
            hop_scores.sort(key=lambda x: x[1], reverse=True)
            results["relevant_hops"] = [hop for hop, score in hop_scores[:top_k] if score > 0]

            # 如果没有相关性，至少返回最近的k个
            if not results["relevant_hops"]:
                results["relevant_hops"] = self.reasoning_chain[-top_k:]

        # 3. 检索相关的事实
        if self.accumulated_facts:
            # 简单策略：返回所有事实（通常不多）
            results["relevant_facts"] = self.accumulated_facts.copy()

        # 4. 检索相关的检索历史
        if self.search_history:
            # 返回最近的k次检索
            recent_searches = dict(list(self.search_history.items())[-top_k:])
            results["relevant_searches"] = recent_searches

        return results

    def _embed_text(self, text: str) -> list:
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        embedding = self._embedding_func.embed_query(text)
        self._embedding_cache[text] = embedding
        return embedding

    def _retrieve_relevant_vector(self, current_query: str, top_k: int = 3) -> Dict[str, Any]:
        results = {
            "relevant_hops": [],
            "relevant_facts": {},
            "relevant_searches": {},
            "task_context": None
        }

        query_embedding = self._embed_text(current_query)

        # Task context
        if self.task_status:
            results["task_context"] = {
                "description": self.task_status.task_description,
                "current_hop": self.task_status.current_hop,
                "status": self.task_status.status
            }

        # Hops
        if self.reasoning_chain:
            hop_scores = []
            for hop in self.reasoning_chain:
                hop_text = f"{hop.query} {hop.action} {hop.observation}"
                score = cosine_similarity(query_embedding, self._embed_text(hop_text))
                hop_scores.append((score, hop))
            hop_scores.sort(key=lambda x: x[0], reverse=True)
            results["relevant_hops"] = [hop for _, hop in hop_scores[:top_k]]

            if not results["relevant_hops"]:
                results["relevant_hops"] = self.reasoning_chain[-top_k:]

        # Facts (vector ranked)
        if self.accumulated_facts:
            fact_scores = []
            for key, value in self.accumulated_facts.items():
                fact_text = f"{key}: {value}"
                score = cosine_similarity(query_embedding, self._embed_text(fact_text))
                fact_scores.append((score, key, value))
            fact_scores.sort(key=lambda x: x[0], reverse=True)
            top_facts = fact_scores[:top_k]
            results["relevant_facts"] = {k: v for _, k, v in top_facts}

        # Searches (vector ranked)
        if self.search_history:
            search_scores = []
            for query, result in self.search_history.items():
                score = cosine_similarity(query_embedding, self._embed_text(query))
                search_scores.append((score, query, result))
            search_scores.sort(key=lambda x: x[0], reverse=True)
            top_searches = search_scores[:top_k]
            results["relevant_searches"] = {q: r for _, q, r in top_searches}

        return results

    # ========== 格式化为Prompt ==========

    def format_for_prompt(self, compact: bool = True, retrieved_data: Dict[str, Any] = None) -> str:
        """
        将共享记忆格式化为可注入prompt的字符串

        Args:
            compact: 是否使用紧凑格式（节省token）
            retrieved_data: 检索得到的数据（如果为None则使用全量数据）

        Returns:
            格式化的字符串
        """
        # 使用检索数据或全量数据
        if retrieved_data:
            task_context = retrieved_data.get("task_context")
            relevant_hops = retrieved_data.get("relevant_hops", [])
            relevant_facts = retrieved_data.get("relevant_facts", {})
            relevant_searches = retrieved_data.get("relevant_searches", {})
        else:
            # 兜底：全量读取
            task_context = {"description": self.task_status.task_description,
                          "current_hop": self.task_status.current_hop} if self.task_status else None
            relevant_hops = self.reasoning_chain[-3:]
            relevant_facts = self.accumulated_facts
            relevant_searches = dict(list(self.search_history.items())[-3:])

        if not task_context and not relevant_hops and not relevant_facts:
            return ""

        lines = ["【共享记忆 - 所有Agent可见】"]

        # 1. 任务状态
        if task_context:
            lines.append(f"任务: {task_context['description'][:80]}")
            if 'current_hop' in task_context:
                total_hops = self.task_status.total_hops_estimated if self.task_status else 2
                lines.append(f"进度: Hop {task_context['current_hop']}/{total_hops}")

        # 2. 推理链（检索到的相关hops）
        if relevant_hops:
            lines.append("\n推理链:")
            for hop in relevant_hops:
                status = "✓" if hop.success else "✗"
                if compact:
                    lines.append(f"  Hop{hop.hop_id} {status}: {hop.action} → {hop.observation[:50]}")
                else:
                    lines.append(f"  Hop {hop.hop_id} {status}:")
                    lines.append(f"    Query: {hop.query}")
                    lines.append(f"    Action: {hop.action}")
                    lines.append(f"    Result: {hop.observation[:100]}")
                    if hop.extracted_answer:
                        lines.append(f"    Answer: {hop.extracted_answer}")

        # 3. 已累积的事实
        if relevant_facts:
            lines.append("\n已知事实:")
            for key, value in list(relevant_facts.items())[:5]:  # 最多5个
                lines.append(f"  • {key}: {value}")

        # 4. 已检索内容（去重提示）
        if relevant_searches:
            lines.append("\n已检索内容（避免重复）:")
            for query in list(relevant_searches.keys()):
                lines.append(f"  • {query}")

        return "\n".join(lines)

    def get_compact_summary(self) -> str:
        """获取超紧凑摘要（token受限时使用）"""
        parts = []

        if self.task_status:
            parts.append(f"Task: {self.task_status.task_description[:40]}")

        if self.reasoning_chain:
            last_hop = self.reasoning_chain[-1]
            parts.append(f"Last: {last_hop.action} → {last_hop.observation[:30]}")

        if self.accumulated_facts:
            facts_str = ", ".join([f"{k}={v}" for k, v in list(self.accumulated_facts.items())[:2]])
            parts.append(f"Facts: {facts_str}")

        return " | ".join(parts)

    # ========== 辅助方法 ==========

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息（用于分析）"""
        return {
            "total_hops": len(self.reasoning_chain),
            "successful_hops": sum(1 for hop in self.reasoning_chain if hop.success),
            "failed_hops": sum(1 for hop in self.reasoning_chain if not hop.success),
            "unique_searches": len(self.search_history),
            "accumulated_facts": len(self.accumulated_facts),
            "agent_interactions": len(self.agent_interactions),
            "task_status": self.task_status.status if self.task_status else "unknown"
        }

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典（用于持久化/调试）"""
        return {
            "reasoning_chain": [
                {
                    "hop_id": hop.hop_id,
                    "query": hop.query,
                    "action": hop.action,
                    "observation": hop.observation,
                    "extracted_answer": hop.extracted_answer,
                    "agent": hop.agent,
                    "success": hop.success,
                    "timestamp": hop.timestamp
                }
                for hop in self.reasoning_chain
            ],
            "search_history": self.search_history,
            "accumulated_facts": self.accumulated_facts,
            "task_status": {
                "task_description": self.task_status.task_description,
                "current_hop": self.task_status.current_hop,
                "status": self.task_status.status,
                "final_answer": self.task_status.final_answer
            } if self.task_status else None,
            "statistics": self.get_statistics()
        }


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 创建共享记忆（单例）
    shared = SharedMemory()

    # 初始化任务
    shared.reset(task_description="Who is the spouse of the director of Titanic?")

    # Hop 1: Retriever建议，Solver执行
    shared.record_interaction("retriever", "solver", "建议: Search[Titanic director]")
    shared.add_hop(
        query="Who directed Titanic?",
        action="Search[Titanic director]",
        observation="James Cameron directed Titanic (1997).",
        extracted_answer="James Cameron",
        agent="solver"
    )
    shared.add_fact("director", "James Cameron")
    shared.add_search_result("Titanic director", "James Cameron")

    # Hop 2
    shared.add_hop(
        query="Who is James Cameron's spouse?",
        action="Search[James Cameron spouse]",
        observation="James Cameron married Suzy Amis in 2000.",
        extracted_answer="Suzy Amis",
        agent="solver"
    )
    shared.add_fact("spouse", "Suzy Amis")

    # 完成任务
    shared.mark_completed("Suzy Amis")

    # 格式化为prompt
    print(shared.format_for_prompt(compact=True))
    print("\n" + "="*60 + "\n")
    print("Statistics:", shared.get_statistics())
