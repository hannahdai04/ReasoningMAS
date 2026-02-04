import json
import re
from typing import Any, Dict, List, Optional

from mas.llm import Message


class MemoryCurator:
    """
    Lightweight curator that extracts structured memory from (action, observation).
    It only writes memory and does not affect the main reasoning path.
    """

    def __init__(self, llm_model: Optional[Any] = None, max_tokens: int = 256):
        self._llm = llm_model
        self._max_tokens = max_tokens

    def curate(self, task_main: str, action: str, observation: str) -> Dict[str, Any]:
        result = self._empty()

        if not action or not observation:
            return result
        if not (action.startswith("Search[") or action.startswith("Lookup[")):
            return result
        if "Invalid Action" in observation:
            return result

        if self._llm:
            llm_result = self._curate_with_llm(task_main, action, observation)
            if llm_result:
                result = self._normalize(llm_result)

        if not result.get("answer"):
            rule_result = self._curate_with_rules(action, observation)
            result = self._merge(result, rule_result)

        return result

    def _curate_with_llm(self, task_main: str, action: str, observation: str) -> Optional[Dict[str, Any]]:
        prompt = (
            "Extract key information as JSON only.\n"
            "Required keys: answer, entities, facts, relations, next_queries.\n"
            "facts: list of {key, value}. relations: list of {head, relation, tail}.\n"
            f"Task: {task_main}\n"
            f"Action: {action}\n"
            f"Observation: {observation}\n"
        )
        messages = [
            Message("system", "Return only valid JSON. No extra text."),
            Message("user", prompt),
        ]
        try:
            raw = self._llm(messages, temperature=0, max_tokens=self._max_tokens)
        except Exception:
            return None
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try to extract JSON object from mixed output
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
        return None

    def _curate_with_rules(self, action: str, observation: str) -> Dict[str, Any]:
        query = self._extract_query(action)
        obs = observation.replace("\n", " ").strip()
        answer = self._extract_answer(obs)

        facts: List[Dict[str, str]] = []
        relations: List[Dict[str, str]] = []
        next_queries: List[str] = []

        if answer:
            if "director" in query.lower():
                facts.append({"key": "director", "value": answer})
                relations.append({"head": query, "relation": "director", "tail": answer})
                next_queries.append(f"{answer} spouse")
            elif "spouse" in query.lower() or "married" in query.lower():
                facts.append({"key": "spouse", "value": answer})
                relations.append({"head": query, "relation": "spouse", "tail": answer})
            else:
                facts.append({"key": "entity", "value": answer})
                relations.append({"head": query, "relation": "related_to", "tail": answer})

        return {
            "answer": answer,
            "entities": [answer] if answer else [],
            "facts": facts,
            "relations": relations,
            "next_queries": next_queries,
        }

    @staticmethod
    def _extract_query(action: str) -> str:
        match = re.match(r"^\\w+\\[(.+)\\]$", action)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_answer(observation: str) -> Optional[str]:
        patterns = [
            r"directed by ([A-Z][a-zA-Z\\s]+)",
            r"directed ([A-Z][a-zA-Z\\s]+)",
            r"married ([A-Z][a-zA-Z\\s]+)",
            r"spouse is ([A-Z][a-zA-Z\\s]+)",
        ]
        for pat in patterns:
            match = re.search(pat, observation, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _normalize(data: Dict[str, Any]) -> Dict[str, Any]:
        def _list(x):
            return x if isinstance(x, list) else []

        return {
            "answer": data.get("answer"),
            "entities": _list(data.get("entities")),
            "facts": _list(data.get("facts")),
            "relations": _list(data.get("relations")),
            "next_queries": _list(data.get("next_queries")),
        }

    @staticmethod
    def _merge(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        if not base:
            return other
        merged = dict(base)
        # Only dedupe string lists safely; keep dict lists as-is
        for key in ["entities", "next_queries"]:
            b = base.get(key) or []
            o = other.get(key) or []
            merged[key] = list(dict.fromkeys([x for x in b + o if isinstance(x, str)]))
        for key in ["facts", "relations"]:
            b = base.get(key) or []
            o = other.get(key) or []
            merged[key] = b + o
        if not merged.get("answer"):
            merged["answer"] = other.get("answer")
        return merged

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {
            "answer": None,
            "entities": [],
            "facts": [],
            "relations": [],
            "next_queries": [],
        }
