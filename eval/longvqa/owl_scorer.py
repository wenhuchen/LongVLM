from typing import Any, Callable, Optional, Sequence, Union
import re
import editdistance
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice


def remove_special_chars_and_lower(s):
    pattern = r"[^a-zA-Z0-9\s]"
    s = re.sub(pattern, "", s)
    return s.lower()


def exact_accuracy(target: str, prediction: str):
    return float(target == prediction)


def relaxed_accuracy(target: str, prediction: str, max_relative_change: float = 0.05):
    def _to_float(text: str) -> Optional[float]:
        try:
            return float(text.rstrip("%")) / 100.0 if text.endswith("%") else float(text)
        except ValueError:
            return None
        
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return float(relative_change <= max_relative_change)
    else:
        return float(prediction.lower() == target.lower())


def contain_accuracy(target:str, prediction:str):
    def has_word(sentence, word):
        pattern = r"\b" + re.escape(word) + r"\b"
        match = re.search(pattern, sentence)
        return True if match else False
    return float(has_word(prediction, target))


def iou_match(target: list, prediction: list, threshold=0.5):
    g_x1, g_y1, g_x2, g_y2 = target
    p_x1, p_y1, p_x2, p_y2 = prediction
    g_w, p_w, g_h, p_h = g_x2 - g_x1, p_x2 - p_x1, g_y2 - g_y1, p_y2 - p_y1
    
    W = (min(g_x2, p_x2) - max(g_x1, p_x1))
    H = (min(g_y2, p_y2) - max(g_y1, p_y1))
    Intersection = W * H
    if Intersection <= 0:
        return 0.0

    Union = g_w * g_h + p_w * p_h - Intersection
    return float(Intersection / Union >= threshold)


def anls_metric(target: str, prediction: str, theta: float = 0.5):
    edit_distance = editdistance.eval(target, prediction)
    normalized_ld = edit_distance / max(len(target), len(prediction))
    return 1.0 - normalized_ld if normalized_ld < theta else 0.0


def metric_cal(
    targets: Sequence[Sequence[str]],
    predictions: Sequence[str],
    metric_fn: Callable[[str, str], Any],
    normalize_fn: Callable[[str], str] = lambda v: v
):
    assert len(targets) == len(predictions)
    scores = []
    for prediction, target in zip(predictions, targets):
        p = normalize_fn(prediction)
        score = max(metric_fn(normalize_fn(t), p) for t in target)
        scores.append(score)
    score = (100.0 * sum(scores)) / len(targets)
    return score, scores


def coco_cal(
    targets: Sequence[Sequence[str]],
    predictions: Sequence[str],
    scorer: Union[Bleu, Meteor, Rouge, Cider, Spice],
    ngram: int = 0
):
    coco_tokenizer = PTBTokenizer()
    score, scores = scorer.compute_score(
    gts=coco_tokenizer.tokenize({
        str(i): [{"caption": t} for t in target]
        for i, target in enumerate(targets)
    }),
    res=coco_tokenizer.tokenize({
        str(i): [{"caption": prediction}]
        for i, prediction in enumerate(predictions)
    }))
        
    if isinstance(scorer, Bleu):
        assert ngram <= 4 and ngram > 0
        score = score[ngram - 1]
        scores = scores[ngram - 1]
    score = float(score) * 100.0
    if not isinstance(scorer, Spice):
        scores = [float(s) * 100.0 for s in scores]
    return score, scores
    

def scorer(
    metric: str,
    targets: Sequence[Sequence[str]],
    predictions: Sequence[str]
):    
    match metric:
        case 'EACC':
            score, scores = metric_cal(targets, predictions, metric_fn=exact_accuracy)
        case 'RACC':
            score, scores = metric_cal(targets, predictions, metric_fn=relaxed_accuracy)
        case 'CACC':
            score, scores = metric_cal(targets, predictions, metric_fn=contain_accuracy, normalize_fn=remove_special_chars_and_lower)
        case 'IOU': 
            score, scores = metric_cal(targets, predictions, metric_fn=iou_match)
        case 'ANLS':
            score, scores = metric_cal(targets, predictions, metric_fn=anls_metric, normalize_fn=lambda v: v.lower())
        case 'BLEU1':
            score, scores = coco_cal(targets, predictions, Bleu(4), 1)
        case 'BLEU2':
            score, scores = coco_cal(targets, predictions, Bleu(4), 2)
        case 'BLEU3':
            score, scores = coco_cal(targets, predictions, Bleu(4), 3)
        case 'BLEU4':
            score, scores = coco_cal(targets, predictions, Bleu(4), 4)
        case 'CIDER':
            score, scores = coco_cal(targets, predictions, Cider())
        case 'ROUGE':
            score, scores = coco_cal(targets, predictions, Rouge())
        case 'METEOR':
            score, scores = coco_cal(targets, predictions, Meteor())
        case 'SPICE':
            score, scores = coco_cal(targets, predictions, Spice())
        case _:
            raise AssertionError
            
    return score, scores 
