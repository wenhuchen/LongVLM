import logging
from typing import Any, Dict, List
from operator import itemgetter

import textdistance

from .base_scorer import BaseScorer

logger = logging.getLogger(__name__)


class AnlsScorer(BaseScorer):
    """ANSL Scorer."""

    def __init__(self, threshold: float = 0.5):
        self.__scores: List[float] = []
        self.threshold = threshold

    @property
    def scores(self):
        return self.__scores

    def add(self, annotations: List[Dict[str, Any]]):
        for annotation in annotations:
            pred, ans = annotation['pr'], annotation['an']
            assert isinstance(pred, list) and isinstance(ans, list)
            assert len(pred) == 1
            best_score = max([textdistance.levenshtein.normalized_similarity(pred[0], an) for an in ans])
            if 1 - self.threshold >= best_score:
                best_score = 0.0            
            self.__scores.append(best_score)

    def score(self) -> float:
        if self.__scores:
            return float(sum(self.__scores) / len(self.__scores) * 100)
        return 0.0

    @classmethod
    def metric_name(cls) -> str:
        return "ANLS"
