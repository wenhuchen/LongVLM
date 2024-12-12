import os
import json
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

from collections import defaultdict
from eval.mm_niah.tools import VQAEval


def remove_trailing_zeros(arr):
    arr = np.asarray(arr)

    last_non_zero_index = np.where(arr != 0)[0][-1] if np.any(arr != 0) else -1

    trimmed_array = arr[:last_non_zero_index + 1] if last_non_zero_index != -1 else arr

    return trimmed_array, last_non_zero_index


def remove_trailing_zero_rows(arr):
    arr = np.asarray(arr)

    last_index = arr.shape[0] - 1
    while last_index >= 0 and np.all(arr[last_index] == 0):
        last_index -= 1

    if last_index == -1:
        return arr, -1

    trimmed_array = arr[:last_index + 1]

    return trimmed_array, last_index


x_bins = [1000, 2000, 4000, 8000, 12000, 16000, 24000, 32000, 40000, 48000, 64000]
y_interval = 0.2
vqa = VQAEval()

context_ranges = [f'{i // 1000}k' for i in x_bins]


def is_correct(answer, response):
    response_orig = response
    response = response.strip('.')
    if isinstance(answer, int):
        if response.isdigit():
            return int(int(response) == answer)

        response = response.lower()
        response = response.replace('the answer is', '')
        response = response.replace('*', '')  # parse **A**
        if response.find('.') != -1:
            response = response.split('.')[0]
            response = response.replace(',', '')
            response = response.strip()
        response = response.strip()

        if response == 'none':
            return 0

        if 'the camera is moving left' in response or response == 'left':
            response = 'a'
        elif 'the camera is moving right' in response or response == 'right':
            response = 'b'

        if len(response) != 1:
            return 0

        return (ord(response) - ord('a')) == answer

    if isinstance(answer, list):
        try:
            response = response.replace('json', '').replace('```', '').strip()
            response = json.loads(response)
            if isinstance(response, dict):
                response = sum(list(response.values()), start=[])
        except Exception as e:
            # print(f"Fail to parse {response_orig} Exception: {e}")
            return 0

        if not isinstance(response, (list, tuple)):
            # print(f"Fail to parse {response_orig} Exception: not a list!")
            return 0

        match = 0
        for res, ans in zip(response, answer):
            match += res == ans
        return match / len(answer)

    else:
        r = deepcopy(response)
        r = r.lower()
        r = r.replace('the answer is', '')
        r = r.replace('*', '')
        if r.find('.') != -1:
            r = r.split('.')[0]
            r = r.replace(',', '')
            r = r.strip()
        r = r.strip()

        a = deepcopy(answer)
        a = a.lower()
        a = a.replace('the answer is', '')
        a = a.replace('*', '')
        if a.find('.') != -1:
            a = a.split('.')[0]
            a = a.replace(',', '')
            a = a.strip()

        if r == a:
            return 1

        return vqa.evaluate(response, answer)


def save(res, weighted_acc, sample_number_array, weighted_avg_acc, save_path):
    res = res.copy()

    overall_scores = []
    for task_name, scores in res.items():
        overall_scores.append(scores)
    overall_scores = np.array(overall_scores).mean(axis=0).tolist()
    # average_scores = np.array(overall_scores).mean(axis=1).tolist()

    if len(res) == 6:
        res['overall'] = [round(item, 6) for item in overall_scores]
    elif len(res) == 1:
        res['average'] = np.mean(overall_scores)
    else:
        print(
            f'[Warning] Since {len(res)=} is not equal to 6, the overall score will be ignored.',
            'Please ensure that you correctly organize the directory structure.'
        )
        print()

    res['weighted_acc'] = weighted_acc.tolist()
    res['sample_number_array'] = sample_number_array.tolist()
    res['weighted_avg_acc'] = weighted_avg_acc

    res['context_ranges'] = context_ranges

    with open(save_path, 'w') as file:
        json.dump(res, file, indent=4)


def main(args):
    res = defaultdict(lambda: defaultdict(dict))

    plt.figure(figsize=(10, 10))
    result_path_list = os.listdir(args.outputs_dir)
    for cur_file_name in result_path_list:
        if cur_file_name.endswith('jsonl'):
            file_name = cur_file_name
            break

    jsonl_file_path = os.path.join(args.outputs_dir, file_name)

    total = np.zeros((len(x_bins) + 1, int(1 / y_interval)))
    correct = np.zeros((len(x_bins) + 1, int(1 / y_interval)))

    model_name, task_name = file_name.replace('.jsonl', '').rsplit('_', 1)
    file_path = os.path.join(args.save_dir, model_name, f'heatmaps_png/{task_name}.png')
    file_path_pdf = os.path.join(args.save_dir, model_name, f'heatmaps_pdf/{task_name}.pdf')

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    os.makedirs(os.path.dirname(file_path_pdf), exist_ok=True)

    with open(jsonl_file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            x = entry['context_length']
            y = entry['placed_depth']
            if isinstance(y, list):
                y = sum(entry['placed_depth']) / len(entry['placed_depth'])
            else:
                y = entry['placed_depth']

            if y == 1.0:
                y = 0.99

            z = entry['response']
            answer = entry['answer']

            if 'counting' in jsonl_file_path and not isinstance(answer, list):
                answer = json.loads(answer)

            x_index = np.digitize(x, x_bins)
            y_index = int(y / y_interval)
            total[x_index][y_index] += 1
            correct[x_index][y_index] += is_correct(answer, z)

        total, last_index = remove_trailing_zero_rows(total)
        if last_index != -1:
            correct = correct[:last_index+1]

        sample_number_array = total.sum(1)[1:]
        correct_number_array = correct.sum(1)[1:]

        weighted_acc = np.divide(correct_number_array, sample_number_array, out=np.zeros_like(correct_number_array),
                                 where=sample_number_array != 0)
        weighted_avg_acc = correct.sum() / total.sum()
        result = np.divide(correct, total, out=np.zeros_like(correct), where=total != 0)

    # # Plot a heatmap for a numpy array:
    uniform_data = result[1:].T

    # Define the custom color map
    from matplotlib.colors import LinearSegmentedColormap

    colors = ["#DC143C", "#FFD700", "#3CB371"]  # Red to Yellow to Green
    n_bins = 100  # Discretizes the interpolation into bins
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # ax = sns.heatmap(uniform_data, vmin=0, vmax=1, cmap=cm)
    ax = sns.heatmap(uniform_data, vmin=0, vmax=1, cmap=cm, cbar=False)

    plt.xticks(ticks=np.arange(uniform_data.shape[1]) + 0.5, labels=[f'{i / 1000}k' for i in x_bins[:last_index]])
    plt.xticks(rotation=45, fontsize=28, fontweight='bold')

    plt.yticks(ticks=np.arange(uniform_data.shape[0] + 1),
               labels=[f'{j / (1 / y_interval)}' for j in range(int(1 / y_interval) + 1)])
    plt.yticks(rotation=0, fontsize=28, fontweight='bold')

    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.savefig(file_path_pdf, dpi=300, bbox_inches='tight')
    plt.clf()

    scores = [round(item, 6) for item in uniform_data.mean(axis=0).tolist()]

    split='val'
    res[model_name][split][task_name] = scores

    save_path = os.path.join(args.save_dir, model_name, f'scores_{split}.json')
    save(res[model_name][split], weighted_acc, sample_number_array, weighted_avg_acc, save_path)
    print(f'results on {split} split of {model_name} are save in {save_path}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation script for MM-NIAH")
    parser.add_argument('--outputs-dir', type=str, default='outputs_example')
    args = parser.parse_args()

    args.save_dir = os.path.join(args.outputs_dir, 'results')
    main(args)
