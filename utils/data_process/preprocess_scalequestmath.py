import json

from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset('dyyyyyyyy/ScaleQuest-Math', split='train')

print("Converting dataset to jsonl format")
output_file = "./ScaleQuest_Math_train_1m.jsonl"
ix = 0
with open(output_file, 'w', encoding='utf-8') as f:
    for item in tqdm(dataset):
        conv = {
            'id': ix,
            'conversations': [
                {'from': 'human', 'value': item['query']},
                {'from': 'gpt', 'value': item['response']}
            ]
        }
        ix += 1
        f.write(json.dumps(conv, ensure_ascii=False) + '\n')

print(f"Conversion complete. Output saved as {output_file}")