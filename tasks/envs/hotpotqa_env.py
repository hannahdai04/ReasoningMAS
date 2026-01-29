from typing import Any, Literal
import re
from dataclasses import dataclass
from typing import Optional, Union

from .base_env import BaseEnv, BaseRecorder
from .utils import match_exactly, f1_score


class HotpotQAEnv(BaseEnv):
    def __init__(
        self,
        env_config: dict[str, Any],
        max_trials: int = 20
    ) -> None:
        self.env_config = env_config
        self.max_trials: int = max_trials
        self.max_sentences_per_doc: int = int(env_config.get('max_sentences_per_doc', 3))

        self.reset()

    def set_env(self, configs: dict) -> tuple[str, str]:
        if configs.get('answer') is None:
            raise ValueError('Please provide the answer for the question.')
        if configs.get('task') is None:
            raise ValueError('The configs dict should have the `task` attribute.')
        if configs.get('context') is None:
            raise ValueError('The configs dict should have the `context` attribute.')

        self.config = configs
        self._build_docs(self.config.get('context'))

        task: str = f"Question: {self.config.get('task')}"
        return task, task

    def reset(self) -> None:
        self.current_task: str = None
        self.reward: float = 0
        self.document = None
        self.lookup_str = ""
        self.lookup_index = 0
        self.docs: list[dict[str, Any]] = []
        self.last_prediction: Optional[str] = None
        self.last_em: Optional[float] = None
        self.last_f1: Optional[float] = None

    def _build_docs(self, context: list[list[Any]]) -> None:
        docs = []
        for item in context:
            if not item or len(item) < 2:
                continue
            title = item[0]
            sentences = item[1]
            if not isinstance(sentences, list):
                continue
            text = " ".join(sentences)
            docs.append({
                'title': title,
                'sentences': sentences,
                'text': text
            })
        self.docs = docs

    def step(self, action: str) -> tuple[str, float, bool]:
        action = self.process_action(action)

        if self._parse_action_type(action) == 'thought':
            return 'OK.', 0, False

        action_type, argument = self._parse_action(action)

        if action_type == 'Finish':
            self.last_prediction = argument
            self.last_em = 1.0 if match_exactly(argument, self.config.get('answer')) else 0.0
            self.last_f1 = f1_score(argument, self.config.get('answer'))
            if self.success_fn(argument):
                observation = 'Answer is CORRECT'
                self.reward = 1
                return observation, 1, True
            observation = 'Answer is INCORRECT'
            return observation, 0, True

        if action_type == 'Search':
            observation = self._search(argument)
        elif action_type == 'Lookup':
            observation = self._lookup(argument)
        else:
            observation = 'Invalid Action. Valid Actions are Lookup[<keyword>] Search[<query>] and Finish[<answer>].'

        if 'Invalid Action' in observation:
            processed_reward = -1
        else:
            processed_reward = 0
        return observation, processed_reward, False

    @staticmethod
    def _parse_action_type(action: str) -> Literal['action', 'thought']:
        if 'thought' in action.lower():
            return 'thought'
        return 'action'

    @staticmethod
    def process_action(action: str) -> str:
        action = action.strip().replace('<', '').split('\n')[0]
        action = action.replace('>', '').replace('OK.', '').replace('OK', '').strip()
        if HotpotQAEnv._parse_action_type(action) == 'thought':
            return action
        if ':' in action:
            action = action.split(':')[1].strip()
        return action

    @staticmethod
    def _parse_action(string: str) -> tuple[str, str]:
        pattern = r'^(\w+)\[(.+)\]$'
        match = re.match(pattern, string)
        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument
        return None, None

    def _search(self, query: str) -> str:
        query = (query or '').strip()
        if not query:
            return 'Invalid Action. Valid Actions are Lookup[<keyword>] Search[<query>] and Finish[<answer>].'

        best_doc = None
        best_score = 0
        query_tokens = set(query.lower().split())

        for doc in self.docs:
            title = doc['title']
            text = doc['text']
            title_tokens = set(title.lower().split())
            text_tokens = set(text.lower().split())
            score = len(query_tokens & title_tokens) * 3 + len(query_tokens & text_tokens)
            if score > best_score:
                best_score = score
                best_doc = doc

        if best_doc is None or best_score == 0:
            titles = [doc['title'] for doc in self.docs[:5]]
            return f"Could not find [{query}]. Similar: {titles}"

        self.document = best_doc
        self.lookup_str = ""
        self.lookup_index = 0

        sentences = best_doc['sentences'][: self.max_sentences_per_doc]
        return f"{best_doc['title']}: {' '.join(sentences)}"

    def _lookup(self, term: str) -> str:
        if self.document is None:
            return 'The last page Searched was not found, so you cannot Lookup a keyword in it. Please Search first.'

        term = (term or '').strip().lower()
        if term != self.lookup_str:
            self.lookup_str = term
            self.lookup_index = 0
        else:
            self.lookup_index += 1

        sentences = self.document.get('sentences', [])
        lookups = [s for s in sentences if term in s.lower()]
        if len(lookups) == 0:
            return 'No Results'
        if self.lookup_index >= len(lookups):
            return 'No More Results'
        result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
        return f"{result_prefix} {lookups[self.lookup_index]}"

    def success_fn(self, agent_ans: str) -> bool:
        return match_exactly(agent_ans, self.config.get('answer'))

    def feedback(self) -> tuple[float, bool, str]:
        feedback: str = 'You successfully finished this task.' if self.reward == 1 else 'You failed the task.'
        done = self.reward == 1
        return self.reward, done, feedback

    def get_metrics(self) -> dict[str, Optional[Union[float, str]]]:
        return {
            'em': self.last_em,
            'f1': self.last_f1,
            'prediction': self.last_prediction
        }


@dataclass
class HotpotQARecorder(BaseRecorder):
    def __post_init__(self):
        super().__post_init__()
        self.task = 'hotpotqa'
        self.counts = 0
        self.dones = 0
        self.rewards = 0
        self.em_sum = 0.0
        self.f1_sum = 0.0

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)
        message: str = f'---------- Task: {task_id} ----------'
        self.log(message)

    def task_end(self, reward: float, done: bool, **kwargs):
        self.rewards += reward
        self.dones += done
        self.counts += 1

        em = kwargs.get('em')
        f1 = kwargs.get('f1')
        if em is None:
            em = 1.0 if done else 0.0
        if f1 is None:
            f1 = 1.0 if done else 0.0

        self.em_sum += em
        self.f1_sum += f1

        message = (
            f'reward: {reward}, ave reward: {self.rewards / self.counts}.\n'
            f'done: {done}, ave done: {self.dones / self.counts}\n'
            f'em: {em:.3f}, ave em: {self.em_sum / self.counts:.3f}\n'
            f'f1: {f1:.3f}, ave f1: {self.f1_sum / self.counts:.3f}'
        )
        self.log(message)
