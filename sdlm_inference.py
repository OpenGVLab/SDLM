import torch
import torch.nn.functional as F
import torch.distributions as dists
import time
from datetime import datetime
from typing import List, Union, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

EOS_TOKEN_IDS = [151645, 151643]
DEFAULT_MASK_TOKEN_ID = 151665
PAD_TOKEN_ID = 151643


def top_p_logits(
    logits: torch.Tensor, 
    top_p: bool=None
) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(
    logits: torch.Tensor, 
    top_k: bool=None
) -> torch.Tensor:
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(
    logits: torch.Tensor, 
    temperature: float=0, 
    top_p: float=None, 
    top_k: float=None, 
    entropy_conf: bool=False):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1) # get the probs of the token sampled
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if entropy_conf:
        entropy = - torch.sum(probs * torch.log(probs + 1e-9), dim=-1) # (batch_size, seq_len)
        max_entropy = torch.log(torch.tensor(probs.shape[-1], dtype=torch.float32)).to(entropy.device)
        normalized_entropy = 1.0 - (entropy / max_entropy)
        confidence = normalized_entropy

    return confidence, x0


def find_longest_sequence(
    confidence: torch.Tensor, 
    threshold: float
) -> torch.Tensor:
    batch_size, seq_len = confidence.shape
            
    mask = confidence > threshold  
    temp = (~mask).int()  # (batch_size, seq_len)
    first_false_idx = torch.argmax(temp, dim=1)
            
    all_true = mask.all(dim=1)  # (batch_size,)
    lengths = torch.where(all_true, torch.tensor(seq_len, device=confidence.device), first_false_idx)

    return lengths


def SDLM_generate(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    model_inputs: Union[torch.Tensor, dict], 
    max_gen_len: int = 1024,
    temperature: float = 0, 
    top_k: Union[int, None] = None, 
    top_p: Union[float, None] = None, 
    threshold: float = 0.9, 
    n_future_tokens: int = 4,
    only_first_token_keep: bool = False,
    use_cache: bool = True,
    alg: str = 'prob_conf', # 
    save_history: bool = False,
) -> List[str]:
    if n_future_tokens == 1:
        # autoregession decode
        return model.generate(
            **model_inputs,
            max_new_tokens=max_gen_len
        )

    batch_size, seq_len = model_inputs.input_ids.shape
    assert batch_size == 1, 'only batch size = 1 is supported now'
    assert alg in ['prob_conf', 'entropy_conf', 'self_speculative']

    sample_kwargs = {
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'entropy_conf': (alg == 'entropy_conf')
    }
    MASK_TOKEN_ID = getattr(model.config, 'text_mask_token_id', DEFAULT_MASK_TOKEN_ID)
    model.model.block_size = getattr(model.config, 'block_size', n_future_tokens)
    model.model.causal_attn = getattr(model.config, 'causal_attn', False)
    model.model.text_mask_token_id = MASK_TOKEN_ID
    model.use_cache = use_cache

    generated  = model_inputs.input_ids.clone()
    total_generate_len = min(tokenizer.model_max_length, seq_len + max_gen_len)

    print(f'\n{alg=}\n'
        f'with kv cache {use_cache=} {MASK_TOKEN_ID=} '
        f'{max_gen_len=} {total_generate_len=} '
        f'{temperature=} {top_k=} {top_p=} '
        f'{threshold=} '
        f'{n_future_tokens=} '
        f'{only_first_token_keep=}\n======\n\n')

    if alg == 'self_speculative':
        return self_speculative_generate(
            model,
            tokenizer,
            model_inputs,
            sample_kwargs,
            max_gen_len,
            n_future_tokens,
            save_history=save_history        
        )

    history_record = []
    iter_round = 0
    past_key_values = None
    
    timers = {
        'total_start': time.time(),
        'prefill_start': time.time(),
        'prefill_end': None
    }

    while generated.size(1) < total_generate_len:
        iter_round += 1

        mask_tokens = torch.full((batch_size, n_future_tokens-1), MASK_TOKEN_ID, dtype=generated.dtype, device=generated.device)
        if use_cache:
            generated_with_mask = torch.cat(
                (
                    generated, 
                    generated[:, -1].unsqueeze(1),
                    mask_tokens
                ), 
                dim=1
            ) # [batch_size, seq_len + 1 +  n_future_tokens - 1]

            start_idx = past_key_values[0][0].size(2) if past_key_values is not None else 0
            position_ids = torch.arange(
                start_idx, 
                generated_with_mask.size(1), 
                device=generated.device).unsqueeze(0)
            # update pe for kvcache
            position_ids[0, -n_future_tokens:] -= 1

            prepare_inputs =  model.prepare_inputs_for_generation(
                generated_with_mask,
                past_key_values,
                None,
                use_cache = True, # Note
                position_ids = position_ids
            )

            with torch.no_grad():
                outputs = model(
                    **prepare_inputs
                )

            past_key_values = outputs.past_key_values

            # update kvcache
            past_key_values = tuple(
                [
                    (item[0][:, :, :generated.shape[1], :], item[1][:, :, :generated.shape[1], :]) 
                    for item in past_key_values
                ]
            )

        else:
            generated_with_mask = torch.cat(
                (
                    generated, 
                    mask_tokens
                ), 
                dim=1
            ) # [batch_size, seq_len +  n_future_tokens - 1]

            with torch.no_grad():
                outputs = model(
                    input_ids=generated_with_mask,
                    attention_mask=torch.ones_like(generated_with_mask),
                    use_cache=False,
                )

        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        next_token_logits = logits[:, -n_future_tokens:, :] # [batch_size, n_future_tokens, vocab_size]
        
        confidence, x0 = sample_tokens(next_token_logits, **sample_kwargs)

        if only_first_token_keep:
            accept_count = 1
        else:
            accept_count = find_longest_sequence(confidence, threshold)[0].item()
            accept_count = max(1, min(accept_count, n_future_tokens))


        new_tokens = x0[0, :accept_count]
        eos_found = False
        for i, token in enumerate(new_tokens):
            if token.item() in EOS_TOKEN_IDS:
                new_tokens = new_tokens[:i]
                eos_found = True
                break

        generated = torch.cat([generated, new_tokens.unsqueeze(0)], dim=1)

        
        if timers['prefill_end'] is None:
            timers['prefill_end'] = time.time()

        if save_history:
             history_record.append(
                (
                    tokenizer.batch_decode(generated[:, seq_len:]),
                    generated[:, seq_len:].size(1)
                )
            )

        if eos_found:
              break
        
    timers['total_end'] = time.time()
    generated_ids = generated[:, seq_len:]

    print(f"{iter_round=}, generated_token_num={generated_ids.size(1)}\n"
          f"generate time, {timers['total_end'] - timers['total_start']}\n"
          f"prefill time, {timers['prefill_end'] - timers['prefill_start']}\n"
          f"decode time, {timers['total_end'] - timers['prefill_end']}\n")    

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

    if save_history:
        return response, history_record

    return response


def self_speculative_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    model_inputs: Union[str, torch.Tensor, Dict[str, torch.Tensor]],
    sample_kwargs: Dict[str, Any],
    max_gen_len: int = 1024,
    n_future_tokens: int = 4,
    save_history: bool = False,
) -> List[str]:
    batch_size, seq_len = model_inputs.input_ids.shape

    MASK_TOKEN_ID = getattr(model.config, 'text_mask_token_id', DEFAULT_MASK_TOKEN_ID)
    model.model.block_size = getattr(model.config, 'block_size', n_future_tokens)
    model.model.causal_attn = getattr(model.config, 'causal_attn', False)
    model.model.text_mask_token_id = MASK_TOKEN_ID
    model.use_cache = False
    
    iter_round = 1
    total_generate_len = min(tokenizer.model_max_length, seq_len + max_gen_len)
    generated  = model_inputs.input_ids.clone()
    history_record = []

    timers = {
        'total_start': time.time(),
        'total_end': None
    }
    
    def prepare_verify_map(x_out):
        pad_y = torch.full((n_future_tokens, n_future_tokens * 2 - 1), PAD_TOKEN_ID)
        for ix in range(n_future_tokens):
            pad_y[ix, :ix + 1] = x_out[:ix + 1]
            pad_y[ix, ix + 1:ix + n_future_tokens] = MASK_TOKEN_ID
        return pad_y

    # prefill stage
    mask_tokens = torch.full((batch_size, n_future_tokens - 1), MASK_TOKEN_ID, dtype=generated.dtype, device=generated.device)
    generated_with_mask = torch.cat(
        (
            generated,
            mask_tokens
        ),
        dim=1
    )  # [1, seq_len + n_future_tokens - 1]

    with torch.no_grad():
        outputs = model(
            input_ids=generated_with_mask,
            attention_mask=torch.ones_like(generated_with_mask),
            use_cache=False, # TODO
        )

    next_token_logits = outputs.logits[:, -n_future_tokens:, :]  # 1, n_future_tokens, vocab_size

    _, candidate_x0 = sample_tokens(next_token_logits, **sample_kwargs)
    
    todo_verify_y0 = candidate_x0

    # verify stage
    while generated.size(1) < total_generate_len:
        iter_round += 1
        pad_verify_y0 = prepare_verify_map(todo_verify_y0[0]) 
        pad_verify_generated = torch.cat(
            (
                generated.repeat(n_future_tokens, 1),
                pad_verify_y0.to(device=generated.device, dtype=generated.dtype)
            ),
            dim=1
        ) # [1 * n_feature_tokens, seq_len + n_feature_tokens * 2 - 1]
        attention_mask = torch.ones(pad_verify_generated.shape, dtype=generated.dtype, device=generated.device)
        attention_mask[pad_verify_generated == PAD_TOKEN_ID] = 0
        
        with torch.no_grad():
            outputs = model(
                input_ids=pad_verify_generated,
                attention_mask=attention_mask,
                use_cache=False,
            )
        pad_verify_logits = outputs.logits[:, -(n_future_tokens*2-1):, :]  # [1*n_feature_tokens, n_feature_tokens * 2 - 1]
        
        _, pad_verify_y1 = sample_tokens(pad_verify_logits, **sample_kwargs)
        
        accept_count = 1
        for ix in range(1, n_future_tokens):
            if pad_verify_y0[ix, ix] == pad_verify_y1[ix-1, ix-1]:
                accept_count += 1
            else:
                break
        new_tokens = todo_verify_y0[0][:accept_count]

        eos_found = False
        for i, token in enumerate(new_tokens):
            if token.item() in EOS_TOKEN_IDS:
                new_tokens = new_tokens[:i]
                eos_found = True
                break
        
        generated = torch.concat([generated[0], new_tokens], dim=-1).unsqueeze(0)
        if eos_found:
            break
            
        if save_history:
            history_record.append(
                (
                    tokenizer.batch_decode(generated[:, seq_len:]),
                    generated[:, seq_len:].size(1)
                )
            )
 
        todo_verify_y0 = pad_verify_y1[accept_count-1, accept_count-1 : accept_count-1 + n_future_tokens].unsqueeze(0)
        
    
    generated_ids = generated[:, seq_len:]
    timers['total_end'] = time.time()

    print(f'{iter_round=}, generated_token_num={len(generated_ids[0])}\n'
          f"generate time, {timers['total_end'] - timers['total_start']}\n")
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

    if save_history:
        return response, history_record

    return response



if __name__ == "__main__":
    try:
        import flash_attn
    except Exception:
        import sys
        from pathlib import Path

        print('Error, flash_attn is not installed.')
        flash_attn_path = Path("/tmp/flash_attn")
        flash_attn_path.mkdir(exist_ok=True)


        init_file = flash_attn_path / "__init__.py"
        init_file.write_text("""
def flash_attn_func(*args, **kwargs):
    raise ImportError("flash_attn is disabled")
        
def flash_attn_varlen_func(*args, **kwargs):
    raise ImportError("flash_attn is disabled")

class FlashAttention:
    def __init__(self, *args, **kwargs):
        raise ImportError("flash_attn is disabled")
""")

        sys.path.insert(0, str(flash_attn_path.parent))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt_hf = 'shell/playground/train_states/SDLM_3B_D4'

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_hf, 
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map='auto',
        torch_dtype = torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt_hf)

    n_feature_tokens = model.config.block_size

    prompt = 'Write a Fibonacci function in Python.'
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    response, history = SDLM_generate(
        model,
        tokenizer,
        model_inputs,
        max_gen_len = 128,
        temperature = 0,
        threshold = 0.5,
        n_future_tokens = n_feature_tokens,
        alg = 'prob_conf', #  prob_conf | entropy_conf | self_speculative
        save_history = True,
        use_cache = True
    )

    print(response[0])

    print('=======histroy')
    for item in history:
        print('cur total token ', item[1])
        print(item[0][0])
        print('--------')
