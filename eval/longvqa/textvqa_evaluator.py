from typing import List, Dict, Any
from textvqa_scorer import TextVQAScorer


def parser_line(line):
    def rectify(s: str):
        s = s.replace('\n', '').strip()
        if s.endswith('.'):
            s = s[:-1]
        return s
            
    id = line['id']
    le = line['le']
    pr = rectify(line['pr'])
    an = line['an']
    if not isinstance(an, list):
        an = [an]
    an = [rectify(a) for a in an]
    return id, le, pr, an


class TextVQAEvaluator:
    def __init__(
        self,
        annotations: List[Dict[str, Any]],
        metric: str,
    ):
        self.annotations = annotations
        self.metric = metric
        self.__scorer = self.eval_textvqa

    
    def evalute(self):
        return self.metric, self.__scorer()
    
    
    def eval_textvqa(self):
        format_annotation = []
        for annotation in self.annotations:
            id, le, pr, an = parser_line(annotation)
            format_annotation.append({
                "pred_answer": pr,
                "gt_answers": an
            })
        scorer = TextVQAScorer()
        return scorer.score(format_annotation)
