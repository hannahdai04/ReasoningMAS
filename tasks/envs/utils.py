from typing import Optional, Union
from collections import Counter
from langchain.docstore.document import Document
import string
import re
import wikipedia

class LangChainWiki:

    def __init__(self) -> None:
        self.document: Optional[Document] = None
        self.lookup_str = ""
        self.lookup_index = 0

    def search(self, search: str) -> Union[str, Document]:
        try:
            page_content = wikipedia.page(search).content
            url = wikipedia.page(search).url
            result: Union[str, Document] = Document(
                page_content=page_content, metadata={"page": url}
            )
        except wikipedia.PageError:
            result = f"Could not find [{search}]. Similar: {wikipedia.search(search)}"
        except wikipedia.DisambiguationError:
            result = f"Could not find [{search}]. Similar: {wikipedia.search(search)}"
        except Exception:
            result = f"Could not find [{search}]. Similar: {wikipedia.search(search)}"
        
        if isinstance(result, Document):
            self.document = result 
            return self._sumary
        else:
            self.document = None
            return result
    
    def lookup(self, term: str):

        if self.document is None:
            raise ValueError("Cannot lookup without a successful search first")
        if term.lower() != self.lookup_str:
            self.lookup_str = term.lower() 
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self._paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_index >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"

    @property
    def _sumary(self) -> str:
        return self._paragraphs[0]
    
    @property
    def _paragraphs(self) -> list[str]:
        if self.document is None:
            raise ValueError("Cannot get paragraphs without a document")
        return self.document.page_content.split("\n\n")
    


def normalize_answer(s: str):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def match_exactly(answer, key) -> bool:

    n_answer = normalize_answer(answer)
    n_key = normalize_answer(key)
    return n_answer == n_key


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)
