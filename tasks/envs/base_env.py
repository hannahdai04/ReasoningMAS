from dataclasses import dataclass, field
import os
import logging
import time
from abc import ABC, abstractmethod

from mas.agents import Env

class BaseEnv(Env, ABC):
     
    def __init__(self, env_config: dict, max_trials: int):
        pass
    
    @abstractmethod
    def set_env(self, task_config: dict) -> tuple[str, str]:
        pass
    
    @abstractmethod
    def step(self, action: str) -> tuple[str, float, bool]:
        pass

    @classmethod
    @abstractmethod
    def process_action(cls, action: str) -> str:
        pass
    
    @abstractmethod
    def feedback(self) -> tuple[float, bool, str]:
        pass


@dataclass
class BaseRecorder:

    working_dir: str = None
    namespace: str = None
    task: str = None

    def __post_init__(self):

        self.file_path = os.path.join(self.working_dir, self.namespace + '.log')
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True) 
        self.prompt_logs_dir = os.path.join(self.working_dir, "prompt_logs")
        os.makedirs(self.prompt_logs_dir, exist_ok=True)

        self.logger = logging.getLogger(self.namespace)
        self.logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(self.file_path)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(console_handler)

        self.current_task_id: int = None
        self.current_task_config: dict = field(default_factory=dict)

    def task_begin(self, task_id: int, task_config: dict) -> None:
        
        self.current_task_id = task_id
        self.current_task_config = task_config

    def task_end(self, reward: float, done: bool, **kwargs) -> None:
        
        if self.current_task_id is None or self.current_task_config is None:
            raise RuntimeError('The task id or the task config should not be None.')
        
    def dataset_begin(self) -> None:
        
        self.start_time = time.time()

        message: str = f"=============== Task Begin ==============="
        self.log(message)

    def dataset_end(self) -> None:
        message: str = f"=============== Task End ==============="

        end_time = time.time()  
        total_time = end_time - self.start_time
        time_message: str = f"Total execution time: {total_time:.2f} seconds"
        self.log(message)
        self.log(time_message)
    
    def log(self, message: str) -> None:    

        if isinstance(message, str) and message.startswith("[PromptLog]"):
            self._log_prompt(message)
            return
        if hasattr(self, "logger") and self.logger:
            self.logger.info(message)
        else:
            print("Logger is not initialized.")

    def _prompt_log_path(self) -> str:
        task_id = self.current_task_id if self.current_task_id is not None else -1
        return os.path.join(self.prompt_logs_dir, f"{self.namespace}_task_{task_id:04d}.log")

    def _log_prompt(self, message: str) -> None:
        path = self._prompt_log_path()
        is_new = not os.path.exists(path)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(path, "a", encoding="utf-8") as f:
            if is_new:
                task_text = None
                if isinstance(self.current_task_config, dict):
                    task_text = self.current_task_config.get("task") or self.current_task_config.get("task_main")
                f.write(f"=== Task {self.current_task_id} ===\n")
                if task_text:
                    f.write(f"Task Input: {task_text}\n")
                f.write("\n")
            f.write(f"{timestamp}\n{message.strip()}\n\n")
