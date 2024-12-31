import logging
from typing import Any, Dict, List

from .base_scorer import BaseScorer

logger = logging.getLogger(__name__)


class AccuracyScorer(BaseScorer):
    """Accuracy Scorer."""

    def __init__(self,):
        self.__scores: List[float] = []

    @property
    def scores(self):
        return self.__scores

    def check_denotation(self, ans: list, pred: list) -> bool:
        for a in ans:
            if a in pred:
                return True
        return False

    def add(self, annotations: List[Dict[str, Any]]):
        for annotation in annotations:
            pred, ans = annotation['pr'], annotation['an']
            assert isinstance(pred, list) and isinstance(ans, list)
            score = int(self.check_denotation(pred, ans))
            self.__scores.append(score)

    def score(self) -> float:
        if self.__scores:
            return float(sum(self.__scores) / len(self.__scores) * 100)
        return 0.0

    @classmethod
    def metric_name(cls) -> str:
        return "Accuracy"
