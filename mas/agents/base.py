from typing import TypeVar, Optional, Dict, Any, Callable

from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.llm import Message
from mas.agents.private_memory import AgentPrivateMemory
from mas.agents.shared_memory import SharedMemory

T = TypeVar("T")


class Agent:
    def __init__(
        self,
        name: str,
        role: str,
        system_instruction: str,
        reasoning_module: ReasoningBase,
        memory_module = None,
        enable_private_memory: bool = True,
        private_memory_config: Optional[Dict[str, Any]] = None,
        log_fn: Optional[Callable[[str], None]] = None
    ):
        if reasoning_module is None:
            raise ValueError("The reasoning module should not be none.")

        # basic info
        self.name: str = name
        self.profile: str = role
        self.system_instruction: str = system_instruction

        # reasoning module
        self.reasoning: ReasoningBase = reasoning_module
        self.memory = memory_module

        # private memory (单次任务内的agent私有记忆)
        self.private_memory: Optional[AgentPrivateMemory] = None
        if enable_private_memory:
            pm_config = private_memory_config or {}
            self.private_memory = AgentPrivateMemory(
                agent_name=name,
                role=role,
                max_interaction_history=pm_config.get('max_interaction_history', 10),
                max_evidence=pm_config.get('max_evidence', 50),
                max_hypotheses=pm_config.get('max_hypotheses', 10)
            )

        self.total_system_instruction: str = self.system_instruction
        self.log_fn = log_fn
        self._response_count = 0

    def add_task_instruction(self, task_instruction: str) -> str:
        self.total_system_instruction = self.system_instruction + '\n' + task_instruction
        return self.total_system_instruction

    def _emit_log(self, message: str) -> None:
        if self.log_fn:
            self.log_fn(message)

    @staticmethod
    def _short(text: str, max_len: int = 200) -> str:
        if text is None:
            return ""
        text = text.replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _summarize_private(self, retrieved: Dict[str, Any], max_items: int = 3) -> str:
        if not retrieved:
            return "None"
        lines = []
        reasoning = retrieved.get("relevant_reasoning", [])
        evidence = retrieved.get("relevant_evidence", [])
        hypotheses = retrieved.get("relevant_hypotheses", [])
        interactions = retrieved.get("recent_interactions", [])
        if reasoning:
            lines.append("Reasoning:")
            for step in reasoning[:max_items]:
                lines.append(f"- {self._short(step.premise, 80)} -> {self._short(step.conclusion, 80)}")
        if hypotheses:
            lines.append("Hypotheses:")
            for h in hypotheses[:max_items]:
                conf = getattr(h, "confidence", None)
                conf_str = f" (conf={conf:.2f})" if isinstance(conf, (int, float)) else ""
                lines.append(f"- {self._short(h.content, 120)}{conf_str}")
        if evidence:
            lines.append("Evidence:")
            for e in evidence[:max_items]:
                src = getattr(e, "source", "")
                lines.append(f"- {self._short(e.content, 120)} [{self._short(src, 40)}]")
        if interactions:
            lines.append("Interactions:")
            for r in interactions[:max_items]:
                lines.append(f"- {r.peer_name}: {self._short(r.content, 120)}")
        return "\n".join(lines) if lines else "None"

    def _summarize_shared(self, retrieved: Dict[str, Any], max_items: int = 3) -> str:
        if not retrieved:
            return "None"
        lines = []
        hops = retrieved.get("relevant_hops", [])
        facts = retrieved.get("relevant_facts", {})
        searches = retrieved.get("relevant_searches", {})
        if hops:
            lines.append("Hops:")
            for hop in hops[:max_items]:
                lines.append(
                    f"- Hop{hop.hop_id}: {self._short(hop.action, 60)} -> {self._short(hop.observation, 120)}"
                )
        if facts:
            lines.append("Facts:")
            for k, v in list(facts.items())[:max_items]:
                lines.append(f"- {self._short(str(k), 40)} = {self._short(str(v), 80)}")
        if searches:
            lines.append("Searches:")
            for q, r in list(searches.items())[:max_items]:
                lines.append(f"- {self._short(str(q), 80)} -> {self._short(str(r), 80)}")
        return "\n".join(lines) if lines else "None"

    def response(self, user_prompt: str, reason_config: ReasoningConfig, debug: bool = False) -> str:
        """
        Agent响应方法 - 实现完整的 Read → Inject → Act 闭环

        Args:
            user_prompt: 用户输入
            reason_config: 推理配置
            debug: 是否打印调试信息

        Returns:
            Agent的响应
        """
        # ========== (A) Read: 检索相关记忆 ==========
        private_retrieved = None
        shared_retrieved = None

        # 1. 检索私有记忆
        private_prompt = ""
        if self.private_memory:
            private_retrieved = self.private_memory.retrieve_relevant(
                current_query=user_prompt,
                current_state="",
                top_k=3
            )
            if debug:
                print(f"[{self.name}] Private Memory Retrieved: {len(private_retrieved.get('relevant_reasoning', []))} reasoning, {len(private_retrieved.get('relevant_evidence', []))} evidence")

        # 2. 检索共享记忆
        shared_mem = SharedMemory()
        shared_retrieved = shared_mem.retrieve_relevant(
            current_query=user_prompt,
            current_state="",
            top_k=3
        )
        if debug:
            print(f"[{self.name}] Shared Memory Retrieved: {len(shared_retrieved.get('relevant_hops', []))} hops, {len(shared_retrieved.get('relevant_facts', {}))} facts")

        # ========== (B) Inject: 格式化并注入到prompt ==========
        memory_sections = []

        # 先注入共享记忆（所有agent可见）
        shared_prompt = shared_mem.format_for_prompt(
            compact=True,
            retrieved_data=shared_retrieved
        )
        if shared_prompt:
            memory_sections.append(shared_prompt)

        # 再注入私有记忆（仅本agent可见）
        if self.private_memory:
            private_prompt = self.private_memory.format_for_prompt(
                include_role_contract=False,  # 角色信息已在system instruction中
                include_plan=True,
                include_hypotheses=True,
                include_evidence=True,
                include_peers=True,
                include_context=True,
                compact=True,
                retrieved_data=private_retrieved
            )
            if private_prompt:
                memory_sections.append(f"【私有记忆 - 仅{self.name}可见】\n{private_prompt}")

        # 组合最终prompt
        if memory_sections:
            memory_block = "\n\n".join(memory_sections)
            enhanced_prompt = f"{memory_block}\n\n{'='*60}\n{user_prompt}"

            if debug:
                print(f"\n[{self.name}] Enhanced Prompt Preview:")
                print(enhanced_prompt[:500] + "..." if len(enhanced_prompt) > 500 else enhanced_prompt)
                print(f"[{self.name}] Total prompt length: {len(enhanced_prompt)} chars")
        else:
            enhanced_prompt = user_prompt

        # ---------- Logging (readable, structured) ----------
        self._response_count += 1
        private_mode = "vector" if (self.private_memory and getattr(self.private_memory, "_embedding_func", None)) else "keyword"
        shared_mode = "vector" if getattr(shared_mem, "_embedding_func", None) else "keyword"
        approx_tokens = max(1, len(enhanced_prompt) // 4)

        log_lines = [
            f"[PromptLog][Agent:{self.name}][#{self._response_count}] Response Preparation",
            f"- Retrieval Mode: private={private_mode}, shared={shared_mode}",
            f"- Raw User Prompt Chars: {len(user_prompt)}",
        ]

        if self._response_count == 1:
            log_lines.append(f"- System Instruction Preview: {self._short(self.total_system_instruction, 300)}")

        log_lines.append("Shared Memory Retrieved:")
        log_lines.append(self._summarize_shared(shared_retrieved))
        log_lines.append("Private Memory Retrieved:")
        log_lines.append(self._summarize_private(private_retrieved))

        if memory_sections:
            log_lines.append("Injected Shared Memory (preview):")
            log_lines.append(self._short(shared_prompt, 600))
            if self.private_memory:
                log_lines.append("Injected Private Memory (preview):")
                log_lines.append(self._short(private_prompt, 600))

        log_lines.append(f"- Final Prompt Chars: {len(enhanced_prompt)}, approx tokens: {approx_tokens}")
        log_lines.append("Final Prompt Preview:")
        log_lines.append(self._short(enhanced_prompt, 1200))

        self._emit_log("\n".join(log_lines))

        # ========== (C) Act: 调用LLM ==========
        messages: list[Message] = [
            Message('system', self.total_system_instruction),
            Message('user', enhanced_prompt)
        ]
        return self.reasoning(messages, reason_config)



class Env:

    def __init__(self) -> None:
        pass
    
    def set_env(self, configs: dict) -> None:
        pass

    def reset(self) -> None:
        pass

    def step(self, action: str) -> None:
        pass
