from typing import TypeVar, Optional, Dict, Any

from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.llm import Message
from mas.agents.private_memory import AgentPrivateMemory

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
        private_memory_config: Optional[Dict[str, Any]] = None
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

    def add_task_instruction(self, task_instruction: str) -> str:
        self.total_system_instruction = self.system_instruction + '\n' + task_instruction
        return self.total_system_instruction

    def response(self, user_prompt: str, reason_config: ReasoningConfig) -> str:
        # 将私有memory添加到prompt中
        if self.private_memory:
            memory_prompt = self.private_memory.format_for_prompt(
                include_role_contract=False,  # 角色信息已在system instruction中
                include_plan=True,
                include_hypotheses=True,
                include_evidence=True,
                include_peers=True,
                include_context=True,
                max_recent_interactions=3,
                max_recent_reasoning=3,
                max_top_hypotheses=3,
                compact=True  # 使用紧凑格式
            )
            if memory_prompt:  # 只有非空时才添加
                enhanced_prompt = f"{memory_prompt}\n\n---\n{user_prompt}"
            else:
                enhanced_prompt = user_prompt
        else:
            enhanced_prompt = user_prompt

        messages: list[Message] = [Message('system', self.total_system_instruction), Message('user', enhanced_prompt)]
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