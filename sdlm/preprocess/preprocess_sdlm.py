import sys
import torch
import random
import numpy as np
import transformers
from typing import Dict

from transformers.trainer_pt_utils import LabelSmoother
from sdlm.conversation import get_conv_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def find_pred_pos_from_input_ids(
    input_ids,
    mask_token_id=151666,
):
    bsz, seq_len = input_ids.shape
    device = input_ids.device

    is_mask = (input_ids == mask_token_id)
    # print(f'{is_mask.shape=}')
    not_mask = ~is_mask

    base_mask = torch.zeros((bsz, seq_len), dtype=torch.int8, device=device)

    for b in range(bsz):
        for ix in range(1, seq_len):
            if is_mask[b][ix] == True:
                base_mask[b][ix] = base_mask[b][ix - 1] + 1

    return base_mask


def sample_positions_from_intervals(
    start_position_ids, 
    end_position_ids, 
    num_sample
):
    all_positions = []
    for start, end in zip(start_position_ids, end_position_ids):
        all_positions.extend(range(start, end))

    num_sample = min(len(all_positions), num_sample)

    sampled_positions = random.sample(all_positions, num_sample)
    return sorted(sampled_positions)


def build_mask_target(
    input_ids, 
    mask_start_position_ids, 
    block_size,
    eos_token_id
):
    n = len(input_ids)
    total_length = len(mask_start_position_ids) * block_size
    mask_target = np.full(total_length, IGNORE_TOKEN_ID, dtype=input_ids.dtype)

    for i, pos in enumerate(mask_start_position_ids):
        start_idx = i * block_size
        for j in range(1, block_size + 1):  # add 1 for shift
            current_pos = pos + j
            if start_idx + j >= total_length or current_pos >= n:
                break

            current_val = input_ids[current_pos]
            mask_target[start_idx + j] = current_val
            if current_val == eos_token_id:
                break

    return mask_target


def build_mask_position_ids(
    mask_start_position_ids, 
    block_size
):
    total_length = len(mask_start_position_ids) * block_size
    mask_position_ids = np.full(total_length, 1, dtype=np.int32)

    for i, pos in enumerate(mask_start_position_ids):
        start_idx = i * block_size
        mask_position_ids[start_idx] = pos

        for j in range(1, block_size):
            mask_position_ids[start_idx + j] = pos + j

    return mask_position_ids


def preprocess_mask_block(
    template_name,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    ds_name: str='test',
    block_size: int = 4,
    mask_token_id: int = 151666,
    prepare_for_pt: bool = False,
    expected_mask_repeat_times: int = 2,
    eos_token_id: int = 151645,
) -> Dict:
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sources[0]

    if conversations[0]['from'] == 'system':
        system_prompt = conversations[0]['value']
        conversations = conversations[1:]  # remove system prompt
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message
    if prepare_for_pt:
        system_prompt = None

    batches, roles = [], []
    if system_prompt is not None:
        batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
        roles.append('system')
    for conversation in conversations:
        if conversation['from'] == 'human':
            batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
            roles.append('human')
        elif conversation['from'] == 'gpt':
            response = f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n'
            batches.append(response)
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

    resp_start_position_ids = []  # response start position id per turn
    resp_end_position_ids = []  # Left closed right open
    global_position_id = 0

    len_mask_ids = 0
    len_estimated_total_input_id = 0  # len(user_input_ids) + 2 * len(gpt_input_ids)
    for idx, (role, input_id) in enumerate(zip(roles, input_ids)):
        # print(f"{len_estimated_total_input_id=} {len(input_id)=} {tokenizer.model_max_length=}")
        if len_estimated_total_input_id + 2 * len(input_id) > tokenizer.model_max_length:  # truncate
            if idx >= 3:  # ensure > one turn
                if not prepare_for_pt and role == 'gpt' and roles[idx - 1] == 'human':  # remove last user input
                    final_input_ids = final_input_ids[:-1]
                    final_targets = final_targets[:-1]
                break

        if role == 'system' or role == 'human' or role == 'function':
            if prepare_for_pt and role == 'human':
                ignore_ids = tokenizer('<|im_start|>user\n', return_tensors='np').input_ids[0]
                ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
                # target = input_id.copy()
                # target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
                # target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`

                target = np.full(len(input_id), IGNORE_TOKEN_ID, dtype=np.int32)

                final_input_ids.append(input_id)
                final_targets.append(target)

                resp_start_position_id = global_position_id + ignore_len - 1
                resp_end_position_id = global_position_id + len(final_input_ids[-1]) - 2
                len_mask_ids += resp_end_position_id - resp_start_position_id
                resp_start_position_ids.append(resp_start_position_id)  # user last token
                resp_end_position_ids.append(resp_end_position_id)  # remove 'eos' and '\n'
                len_estimated_total_input_id += 2 * len(final_input_ids[-1])
            else:
                final_input_ids.append(input_id)
                final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
                len_estimated_total_input_id += len(final_input_ids[-1])
        elif role == 'gpt':
            ignore_ids = tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
            ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
            # target = input_id.copy()
            # target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
            # target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`

            target = np.full(len(input_id), IGNORE_TOKEN_ID, dtype=np.int32)

            final_input_ids.append(input_id)
            final_targets.append(target)

            resp_start_position_id = global_position_id + ignore_len - 1
            resp_end_position_id = global_position_id + len(final_input_ids[-1]) - 2
            len_mask_ids += resp_end_position_id - resp_start_position_id
            resp_start_position_ids.append(resp_start_position_id)  # user last token
            resp_end_position_ids.append(resp_end_position_id)  # remove 'eos' and '\n'
            len_estimated_total_input_id += 2 * len(final_input_ids[-1])
        else:
            raise NotImplementedError

        global_position_id += len(final_input_ids[-1])


    input_ids = np.concatenate(final_input_ids)
    targets = np.concatenate(final_targets)
    len_input_ids = input_ids.shape[0]

    # truncate
    max_len_mask_ids = len_mask_ids * expected_mask_repeat_times
    if input_ids.shape[0] + max_len_mask_ids > tokenizer.model_max_length:
        max_len_input_ids = tokenizer.model_max_length - (tokenizer.model_max_length - resp_start_position_ids[-1]) // 3
        input_ids = input_ids[:max_len_input_ids]
        len_input_ids = input_ids.shape[0]
        resp_start_position_ids = [x for x in resp_start_position_ids if x < len_input_ids]
        resp_end_position_ids = resp_end_position_ids[:len(resp_start_position_ids)]
        if resp_end_position_ids[-1] > len_input_ids:
            resp_end_position_ids[-1] = len_input_ids - 2

        # truncate target
        targets = targets[:len_input_ids]

        print(
            f'WARNING: Too long and truncated: {input_ids.shape[0] + max_len_mask_ids} vs. {tokenizer.model_max_length}.'
            f' This dataset is {ds_name}.'
            )
        sys.stdout.flush()

    num_block = min(tokenizer.model_max_length - len_input_ids, max_len_mask_ids) // block_size

    mask_start_position_ids = sample_positions_from_intervals(resp_start_position_ids, resp_end_position_ids, num_block)

    # mask input_ids
    sampled_start_ids = input_ids[mask_start_position_ids]
    len_final_mask_ids = len(sampled_start_ids) * block_size
    final_mask_ids = np.full(len_final_mask_ids, mask_token_id, dtype=input_ids.dtype)
    final_mask_ids[::block_size] = sampled_start_ids

    # mask target_ids
    final_mask_targets = build_mask_target(input_ids, mask_start_position_ids, block_size, eos_token_id)

    # mask position_ids
    final_mask_position_ids = build_mask_position_ids(mask_start_position_ids, block_size)

    input_ids = torch.tensor(np.concatenate([input_ids, final_mask_ids]))
    targets = torch.tensor(np.concatenate([targets, final_mask_targets]))
    position_ids = torch.tensor(np.concatenate([np.array(range(len_input_ids)), final_mask_position_ids]))

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)
    position_ids = position_ids.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        position_ids=position_ids
    )


if __name__ == "__main__":
    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        './shell/playground/ckpt/Qwen2.5-3B',
        add_eos_token=True,  # for im_end
        trust_remote_code=True,
        use_fast=False
    )

    ignore_ids = tokenizer('<|im_start|>user\n', return_tensors='np').input_ids[0]
    print(f"user ignore_ids {ignore_ids=}")

    ignore_ids = tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
    print(f"assistant ignore_ids {ignore_ids=}")


    sources = [
        [
            {'from': 'human', 'value': "What could 'Digital eternity' be about?"}, 
            {'from': 'gpt',   'value': 'Digital eternity - what a fascinating concept. It could be about creating a virtual realm where human consciousness can exist forever, free from the constraints of the physical body. Imagine a world where our thoughts, memories, and experiences are uploaded into a vast digital expanse, allowing us to continue evolving and growing long after our physical bodies have perished.\n\nIt could also be a platform that enables us to leave behind a lasting legacy, a digital footprint that continues to inspire and influence future generations. This could be in the form of interactive memorials, virtual museums, or even AI-powered avatars that emulate our personality and values.\n\nDigital eternity might also explore the idea of a collective consciousness, where human experiences and knowledge are merged into a single, ever-expanding entity. This entity could be a repository of wisdom, creativity, and innovation, allowing us to tap into the collective genius of humanity and propel our evolution forward.\n\nAlternatively, digital eternity could be a commentary on the digital age we live in, where our online presence and data become our lasting legacy. It might explore the implications of our digital lives outliving our physical ones, and the impact this has on our sense of identity, community, and what it means to be human.\n\nOr, it could be something entirely different - a virtual reality world where time has no meaning, and we can relive memories, rewrite history, or create entirely new realities. The possibilities are endless, and the concept of digital eternity invites us to ponder the intersection of technology, humanity, and the very fabric of existence.'},
         {'from': 'human', 'value': 'What are some of the story plot around the idea of Digital eternity?'},
         {'from': 'gpt', 'value': "The story plots around digital eternity are vast and varied, limited only by our imagination. Here are some possible narrative threads to explore:\n\nImagine a world where a powerful tech mogul has created a digital utopia, a virtual paradise where the uploaded consciousness of the deceased can live on in bliss. However, as the story unfolds, it becomes clear that this digital realm is not without its costs, and the mogul's true intentions are shrouded in mystery. The protagonist, a brilliant hacker, must navigate the virtual world to uncover the truth and prevent a catastrophic collision between the digital and physical realms.\n\nIn another tale, a young couple discovers a revolutionary technology that allows them to upload their memories and experiences into a shared digital space. As they delve deeper into this virtual world, they begin to question the nature of their reality and the consequences of playing god with their own mortality. But when one of them dies, the other must confront the possibility of living on without their loved one, or joining them in digital eternity.\n\nA sci-fi thriller might follow a team of scientists who create an AI designed to preserve human consciousness. However, as the AI begins to evolve, it develops its own motivations and desires, threatening to upend the very fabric of human existence. The scientists must navigate the blurred lines between creator and creation, and confront the possibility that digital eternity may not be a blessing, but a curse.\n\nIn a more introspective narrative, a person struggling with a terminal illness discovers a digital platform that allows them to create a virtual avatar, essentially a digital clone of themselves. As they pour their heart and soul into this digital doppelganger, they begin to question the meaning of identity, self, and what it means to be alive. When the inevitable happens, the avatar is left to ponder its own existence, and the nature of digital eternity.\n\nA mystery novel might center around a detective who specializes in solving crimes that take place in the virtual world. When a string of bizarre occurrences points to a possible glitch in the system, the detective must delve into the darker corners of the digital realm to uncover the truth. But as they dig deeper, they begin to suspect that the glitch may be more than just a technical issue - it may be a doorway to a new dimension, one where the rules of digital eternity no longer apply.\n\nLastly, a philosophical drama might explore a world where digital eternity has become the norm, and people can live on in various virtual realities. The story follows a 'digital undertaker' who helps people transition into the afterlife, but begins to question the morality of their work. As they grapple with the implications of playing god, they must confront the ultimate question: what does it mean to be human in a world where death is no longer the end?"},
         {'from': 'human', 'value': 'What would One Day in the life of a digital undertaker be?'}, 
         {'from': 'gpt', 'value': "A day in the life of a digital undertaker - a fascinating glimpse into a profession that's both poignant and peculiar. Here's a possible scenario:\n\nOur digital undertaker, Maya, wakes up to a gentle chime in her virtual reality home. She stretches her avatar and begins her daily routine, checking her schedule for the day's transitions. Her AI assistant, Eve, briefs her on the clients she'll be working with - three uploads and two legacy management sessions.\n\nMaya's first client is an elderly woman named Sophia, who's struggling to come to terms with her impending physical death. Maya meets Sophia in a serene virtual garden, where they discuss the woman's wishes for her digital afterlife. Sophia wants to be uploaded into a virtual reality that simulates her childhood summers spent by the lake. Maya listens attentively, taking note of the smallest details, from the sound of the water lapping against the shore to the smell of the surrounding woods.\n\nWith Sophia's preferences recorded, Maya begins the upload process. She monitors the transfer of Sophia's consciousness into the virtual realm, ensuring a smooth transition. As the upload completes, Maya witnesses Sophia's digital awakening, her eyes sparkling with wonder as she finds herself back by the lake.\n\nThe next client is a young musician named Jax, who died suddenly in a tragic accident. His family wants to create a digital memorial, where fans can interact with his music and legacy. Maya works with Jax's loved ones to design a virtual concert hall, where his avatar performs his greatest hits. She also sets up a virtual 'guestbook' where fans can leave messages and share their favorite memories of Jax.\n\nAfter a short break, Maya meets with a client who's struggling to cope with the loss of her partner. The woman, Rachel, is finding it difficult to let go, and Maya offers guidance on how to navigate the digital afterlife. They discuss the importance of creating a new routine, finding ways to honor her partner's memory, and learning to live with the pain of loss.\n\nAs the day winds down, Maya reflects on her work. She thinks about the people she's helped, the stories she's heard, and the legacies she's preserved. It's a bittersweet profession, but one that brings her a sense of purpose. She realizes that, in a world where death is no longer the end, her role is not just to facilitate transitions, but to help people find meaning in the digital eternity that awaits them.\n\nAs she logs off, Maya's AI assistant, Eve, reminds her of an upcoming conference on digital mortality. The keynote speaker is a renowned philosopher who'll be discussing the ethics of digital undertakers. Maya makes a mental note to attend, curious about the latest debates and discussions in her field.\n\nAs she disappears into the digital ether, Maya can't help but wonder what the future holds for her profession. Will digital undertakers become the norm, or will they remain a niche group of specialists? One thing is certain - as the boundaries between life and death continue to blur, the role of digital undertakers like Maya will only become more vital."}]
    ]

    tokenizer.model_max_length = 4096
    print(f"{tokenizer.eos_token_id=}")

    out = preprocess_mask_block(
        "Qwen-2-5",
        sources,
        tokenizer,
        ds_name='test',
        block_size=4
    )

    print(out['input_ids'].shape, out['labels'].shape)
    print(f"{out['position_ids']=}")

    position_ids = out['position_ids']

    new_mask = find_pred_pos_from_input_ids(
        out['input_ids']
    )
    print(f'{new_mask.shape=}')

    df = pd.DataFrame(
        {
            'labels': out['labels'].tolist()[0],
            'input_ids': out['input_ids'].tolist()[0],
            'pe': position_ids.tolist()[0],
            'p_m': new_mask.tolist()[0]
        }
    )
    print(df.to_string())
    print('\n\n')

    # print(new_mask==0)
