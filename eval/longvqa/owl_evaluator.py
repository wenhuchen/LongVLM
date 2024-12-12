from typing import List, Dict, Any
from owl_scorer import scorer


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


class OwlEvaluator:
    def __init__(
        self,
        annotations: List[Dict[str, Any]],
        metric: str,
        dataset: str=''
    ):
        self.annotations = annotations
        self.metric = metric
        if dataset == 'TextCaps':
            self.__scorer = self.eval_textcaps
        elif dataset == 'TextVQA':
            self.__scorer = self.eval_textvqa
        else:
            self.__scorer = self.eval_owl

    
    def evalute(self):
        return self.metric, self.__scorer()
    
    
    def eval_owl(self):
        ids, les, prs, ans = [], [], [], []
        for annotation in self.annotations:
            id, le, pr, an = parser_line(annotation)
            # ids.append(id)
            # les.append(le)
            prs.append(pr)
            ans.append(an)

        score, _ = scorer(metric=self.metric, targets=ans, predictions=prs)
        return score