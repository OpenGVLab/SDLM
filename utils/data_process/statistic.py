import json
from transformers import AutoTokenizer
from tqdm import tqdm

meta_json = './shell/playground/data/meta/sft_opc436k_scale_math_1m_smoltalk_1m_tulu_1m.json'
with open(meta_json, "r", encoding="utf-8") as file:
    meta = json.load(file)

tokenizer = AutoTokenizer.from_pretrained(
    './shell/playground/ckpt/Qwen2.5-3B',
    add_eos_token=True,  # for |im_end|
    trust_remote_code=True,
    use_fast=False
)

token_dict = {}
for data_name, data_item in meta.items():
    annotation = data_item['annotation']
    print(f'processing... {annotation}')
    anno_token_cnt = 0
    with open(annotation, 'r') as f:
        raw_data = f.readlines()
        for data_item in tqdm(raw_data):
            data_item = json.loads(data_item)
            sample = ''
            for conversation in data_item['conversations']:
                sample += conversation['value']
            input_ids = tokenizer(
                [sample],
                return_tensors='np',
                padding=False,
                truncation=False,
            ).input_ids
            anno_token_cnt += input_ids.shape[1]
            
    token_dict[annotation] = anno_token_cnt
    print(f'{annotation} total {anno_token_cnt}')

print(token_dict)

with open("utils/data_process/data_token_stastic.json", "w", encoding="utf-8") as f:
    json.dump(token_dict, f, ensure_ascii=False, indent=4)
