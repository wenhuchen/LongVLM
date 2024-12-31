from typing import List, TypeVar, Dict, Any
from due_scorer import BaseScorer, AccuracyScorer, AnlsScorer, FScorer, WtqScorer


TScorer = TypeVar("TScorer", bound=BaseScorer)


class DueEvaluator:
    def __init__(
        self,
        annotations: List[Dict[str, Any]],
        metric: str,
        ignore_case: bool = True,
    ):
        self.annotations = annotations
        self.metric = metric
        self.ignore_case = ignore_case
        self.scorer = self._create_scorer()


    def _create_scorer(self) -> BaseScorer:
        if self.metric == 'ACC':
            scorer = AccuracyScorer()
        elif self.metric == 'ANLS':
            scorer = AnlsScorer()
        elif self.metric == 'F1':
            scorer = FScorer()
        elif self.metric == 'WTQ':
            scorer = WtqScorer()
        else:
            raise ValueError(self.metric)
        return scorer


    def evalute(self):
        def rectify(s: str):
            return s.strip().rstrip('.')
        
        for annotation in self.annotations:
            assert 'pr' in annotation and 'an' in annotation and 'id' in annotation
            if isinstance(annotation['pr'], str):
                annotation['pr'] = [annotation['pr']]
            if isinstance(annotation['an'], str):
                annotation['an'] = [annotation['an']]
                
            if self.ignore_case:
                annotation['pr'] = [p.lower() for p in annotation['pr']]
                annotation['an'] = [a.lower() for a in annotation['an']]

            annotation['pr'] = [rectify(p) for p in annotation['pr']]
            annotation['an'] = [rectify(a) for a in annotation['an']]
            
        self.scorer.add(self.annotations)
        return self.metric, self.scorer.score()
