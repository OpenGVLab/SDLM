import sys
import torch
import random
import numpy as np
import transformers
from typing import Dict

from transformers.trainer_pt_utils import LabelSmoother
from sdlm.conversation import get_conv_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


"""
data preprocess for sft
"""
def preprocess_text_sft(
    template_name,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    ds_name: str = None,
) -> Dict:
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sources[0]

    if conversations[0]['from'] == 'system':
        system_prompt = conversations[0]['value']
        conversations = conversations[1:]  # remove system prompt
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message

    batches, roles = [], []
    if system_prompt is not None:
        batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
        roles.append('system')
    for conversation in conversations:
        if conversation['from'] == 'human':
            batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
            roles.append('human')
        elif conversation['from'] == 'gpt':
            batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
            roles.append('gpt')
        elif conversation['from'] == 'function':
            batches.append(f'<|im_start|>function\n{conversation["value"]}<|im_end|>\n')
            roles.append('function')
        else:
            raise NotImplementedError

    add_bos_token = getattr(tokenizer, 'add_bos_token', False)
    if add_bos_token:  # for InternLM series
        batches[0] = tokenizer.bos_token + batches[0]

    # Tokenize conversations
    input_ids = tokenizer(
        batches,
        return_tensors='np',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids

    if add_bos_token:  # for InternLM series
        input_ids = [item[1:] for item in input_ids]

    final_input_ids, final_targets = [], []
    ignore_ids = tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
    ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
    for role, input_id in zip(roles, input_ids):
        if role == 'system' or role == 'human' or role == 'function':
            final_input_ids.append(input_id)
            final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
        elif role == 'gpt':
            target = input_id.copy()
            target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
            target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`
            final_input_ids.append(input_id)
            final_targets.append(target)
        else:
            raise NotImplementedError
    input_ids = torch.tensor(np.concatenate(final_input_ids))
    targets = torch.tensor(np.concatenate(final_targets))

    assert len(input_ids) == len(targets)

    current_length = input_ids.size(0)
    if current_length > tokenizer.model_max_length:
        input_ids = input_ids[:tokenizer.model_max_length]
        targets = targets[:tokenizer.model_max_length]
        print(
            f'WARNING: Too long and truncated: {current_length} vs. {tokenizer.model_max_length}.'
            f' This dataset is {ds_name}.'
            )
        sys.stdout.flush()

    # padding = False
    # if padding:
    #     current_length = input_ids.size(0)
    #     padding_length = tokenizer.model_max_length - current_length
    #     input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
    #     targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    # print(f'{conversations=}')
    # print(f'{tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]}')
    # print(f'{batches=}')

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
