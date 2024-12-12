import os
import json
import argparse
from copy import deepcopy

from owl_evaluator import OwlEvaluator
from due_evaluator import DueEvaluator   
from textvqa_evaluator import TextVQAEvaluator
from rectify import rectify


M_DUE = ['ACC', 'F1', 'WTQ']
M_OWL = ['EACC', 'RACC', 'CACC', 'IOU', 'ANLS', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'CIDER', 'ROUGE', 'METEOR', 'SPICE']
M_VQA = ['TEXTVQA']


def read_jsonl(path):
    d = []
    with open(path, 'rb') as f:
        for line in f:
            try:
                d.append(json.loads(line))
            except:
                pass
    return d


def decide_metric(task):
    if task in ['chartqa', 'clevr', 'dvqa', 'gqa', 'ocrvqa']:
        return 'RACC'
    elif task in ['svqa', 'tabfact']:
        return 'EACC'
    elif task in ['deepform', 'kleistercharity']:
        return 'F1'
    elif task in ['docvqa', 'infovqa']:
        return 'ANLS'
    elif task in ['okvqa', 'textvqa', 'vizwiz']:
        return 'TEXTVQA'
    elif task in ['textcaps', 'visualmrc']:
        return 'BLEU4'
    elif task in ['wikitablequestions']:
        return 'WTQ'
    else:
        print(f"Unknown task: {task}, use RACC as default metric.")
        return 'RACC'


def relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05):
    def _to_float(text: str):
        try:
            if text.endswith('%'):
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()

def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)

def main(args):
    annotations = read_jsonl(os.path.join(args.outputs_dir, f'{args.result_file}.jsonl'))
    for a in annotations:
        a['pr'] = rectify(args.task, a['pr'])
        a['an'] = rectify(args.task, a['an'])
    scores = {}
    metrics = [decide_metric(args.task)]
    for metric in metrics:
        annotations_cp = deepcopy(annotations)
        if metric in M_OWL:
            evaluator = OwlEvaluator(annotations_cp, metric)
        elif metric in M_DUE:
            evaluator = DueEvaluator(annotations_cp, metric)
        elif metric in M_VQA:
            evaluator = TextVQAEvaluator(annotations_cp, metric)
        else:
            print("Invalid metric")
            continue
        metric, score = evaluator.evalute()
        scores[metric] = score
        print({"score": score, "metric": metric})
        
    with open(os.path.join(args.outputs_dir, f'{args.score_file}.json'), 'w') as file:
        json.dump(scores, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--outputs-dir', type=str, default='outputs_example')
    parser.add_argument('--result-file', type=str, default='result')
    parser.add_argument('--score-file', type=str, default='score')
    args = parser.parse_args()
    main(args)
