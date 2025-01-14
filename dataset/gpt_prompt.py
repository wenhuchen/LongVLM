import datasets
from openai import OpenAI
import base64
import io
import os
import json
from multiprocessing import Process
import time

client = OpenAI()

def get_response(entry):
    # Convert PIL image to bytes
    images = entry['question_images']
    question = entry['question']
    for i, image in enumerate(images):
        image.save(f'images/{entry["idx"]}_{i}.png')

    # Remove the images because they can be serialized
    del entry['solution_images']
    del entry['question_images']

    # Formulating the prompt
    contents = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}})
    contents.append({"type": "text", "text": question + '\nPlease reason step by step and provide the final answer at the end as `Answer: <answer>`.'})

    count = 0
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # Updated model name
                messages=[
                    {
                        "role": "user",
                        "content": contents
                    },
                ],
                response_format={
                    "type": "text"
                },
                temperature=1,
                max_tokens=2048,  # Updated parameter name
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=4,
            )
            entry['responses'] = [c.message.content for c in response.choices]
            break
        except Exception as e:
            if count > 5:
                entry['responses'] = []
                break
            else:
                print(e)
                time.sleep(4)
            count += 1

    return entry

def process_dataset(dataset, process_id):
    output_file = f'outputs/output_process_{process_id}.json'
    mode = 'a' if os.path.exists(output_file) else 'w'
    handle = open(output_file, mode)

    for entry in dataset:
        response = get_response(entry)
        handle.write(json.dumps(response) + '\n')

    handle.close()
    print(f"Process {process_id}: Finished")


def split_dataset(dataset, num_splits):
    split_size = len(dataset) // num_splits
    splits = [
        dataset.select(range(i * split_size, (i + 1) * split_size)) for i in range(num_splits - 1)
    ]
    splits.append(dataset.select(range((num_splits - 1) * split_size, len(dataset))))
    return splits


if __name__ == "__main__":
    dataset = datasets.load_dataset('TIGER-Lab/VisualWebInstruct2')
    THREADS = 10

    for key in dataset.keys():
        print(key)
        dataset_split = dataset[key]
        print('total length: ', len(dataset_split))
        dataset_split = dataset_split.select(range(50))

        # Distribute the dataset into THREADS chunks
        splits = split_dataset(dataset_split, THREADS)

        # Process each chunk in parallel
        processes = []
        for i, split in enumerate(splits):
            p = Process(target=process_dataset, args=(split, i))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        break
