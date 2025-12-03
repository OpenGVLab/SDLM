import time
import torch
from typing import List, Union, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from functools import partial

EOS_TOKEN_IDS = [151645, 151643] # im_end, end_of_text
DEFAULT_MASK_TOKEN_ID = 151665 # <mask>
PAD_TOKEN_ID = 151643 


class SDLMGenerator:
    def __init__(self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        sampling_args: Dict = None,
        n_future_tokens: int = 4
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sampling_args = sampling_args

        # Initialize additional attributes
        self.MASK_TOKEN_ID = getattr(self.model.config, 'text_mask_token_id', DEFAULT_MASK_TOKEN_ID)
        self.model.model.block_size = getattr(self.model.config, 'block_size', n_future_tokens)
        self.model.model.causal_attn = getattr(self.model.config, 'causal_attn', False)
        self.model.model.text_mask_token_id = self.MASK_TOKEN_ID

        if n_future_tokens != self.model.config.block_size:
            self.model.model.block_size = n_future_tokens

        if self.sampling_args is None:
            self.sampling_args = {
                'temperature': 0,
                'top_p': None,
                'top_k': None,
                'entropy_conf': False
            }


    def top_p_logits(
        self,
        logits: torch.Tensor, 
        top_p: bool=None
    ) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
        mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        return logits

    def top_k_logits(
        self,
        logits: torch.Tensor, 
        top_k: bool=None
    ) -> torch.Tensor:
        top_k = min(top_k, logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        return logits

    def sample_tokens(
        self,
        logits: torch.Tensor, 
        temperature: float=0, 
        top_p: float=None, 
        top_k: float=None, 
        entropy_conf: bool=False
    ):
        if temperature > 0:
            logits = logits / temperature
        if top_p is not None and top_p < 1:
            logits = self.top_p_logits(logits, top_p)
        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)
        probs = torch.softmax(logits, dim=-1)

        if temperature > 0:
            try:
                x0 = torch.distributions.Categorical(probs=probs).sample()
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
        self,
        confidence: torch.Tensor, 
        threshold: float
    ) -> torch.Tensor:
        _, seq_len = confidence.shape
                
        mask = confidence > threshold  
        temp = (~mask).int()  # (batch_size, seq_len)
        first_false_idx = torch.argmax(temp, dim=1)
                
        all_true = mask.all(dim=1)  # (batch_size,)
        lengths = torch.where(all_true, torch.tensor(seq_len, device=confidence.device), first_false_idx)

        return lengths

    def is_eos_found(
        self,
        new_tokens: torch.tensor
    ):
        # eos_mask = torch.isin(new_tokens, torch.tensor(EOS_TOKEN_IDS, device=new_tokens.device))
        # eos_found = eos_mask.any()
        # new_tokens = new_tokens[:eos_mask.nonzero(as_tuple=True)[0].item()] if eos_found else new_tokens
        # return eos_found, new_tokens
        eos_found = False
        for i, token in enumerate(new_tokens):
            if token.item() in EOS_TOKEN_IDS:
                new_tokens = new_tokens[:i]
                eos_found = True
                break
        
        return eos_found, new_tokens

    def generate(
        self, 
        model_inputs: Union[torch.Tensor, dict],
        max_gen_len: int = 1024,     
        use_cache: bool = True,
        alg: str = 'prob_conf', # 
        threshold: float = 0.9,
        save_history: bool = False,
        only_first_token_keep: bool = False,
        n_future_tokens: int = 4
    ) -> List[str]:
        if n_future_tokens == 1:
            # autoregession decode
            return self.model.generate(**model_inputs, max_new_tokens=max_gen_len)

        self.n_future_tokens = n_future_tokens

        batch_size, seq_len = model_inputs.input_ids.shape
        self.total_generate_len = min(self.tokenizer.model_max_length, seq_len + max_gen_len)
        self.save_history = save_history

        assert batch_size == 1, 'only batch size = 1 is supported now'
        assert alg in ['prob_conf', 'entropy_conf', 'self_speculative']

        self.sampling_args['entropy_conf'] == (alg == 'entropy_conf')

        self.timers = {
            'total_start': time.time(),
            'prefill_start': time.time(),
            'prefill_end': None
        }

        print(f'\n-----\n'
              f'generation args {use_cache=} {alg=}\n'
              f'{max_gen_len=} {self.total_generate_len=}\n'
              f'{self.sampling_args=}\n'
              f'{threshold=} {only_first_token_keep=} {n_future_tokens=}\n'
              f'{self.model.training=}'
              f'\n=====\n')

        if alg == 'self_speculative':
            gen_func = self.self_speculative_generate_cached if use_cache else self.self_speculative_generate
        else:
            gen_func = partial(
                self.conf_generate_cached,
                threshold=threshold, only_first_token_keep=only_first_token_keep
            ) if use_cache else partial(
                self.conf_generate,
                threshold=threshold, only_first_token_keep=only_first_token_keep
            )

        generated_ids, history, forward_step = gen_func(model_inputs)

        print(f"{forward_step=}, generated_token_num={generated_ids.size(1)}\n"
            f"generate time, {self.timers['total_end'] - self.timers['total_start']}\n"
            f"prefill time, {self.timers['prefill_end'] - self.timers['prefill_start']}\n"
            f"decode time, {self.timers['total_end'] - self.timers['prefill_end']}\n")    

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        return response, history
    

    def self_speculative_generate(
        self,
        model_inputs: Union[str, torch.Tensor, Dict[str, torch.Tensor]],
    ) -> List[str]:
        batch_size, seq_len = model_inputs.input_ids.shape
        generated  = model_inputs.input_ids.clone()
        
        self.model.use_cache = False

        forward_step = 1
        history_record = []

        def prepare_verify_map(x_out):
            pad_y = torch.full((self.n_future_tokens, self.n_future_tokens * 2 - 1), PAD_TOKEN_ID)
            for ix in range(self.n_future_tokens):
                pad_y[ix, :ix + 1] = x_out[:ix + 1]
                pad_y[ix, ix + 1:ix + self.n_future_tokens] = self.MASK_TOKEN_ID
            return pad_y

        # prefill stage
        generated_with_mask = torch.cat(
            (
                generated,
                torch.full((batch_size, self.n_future_tokens - 1), self.MASK_TOKEN_ID, dtype=generated.dtype, device=generated.device)
            ), dim=1
        )  # [1, seq_len + n_future_tokens - 1]

        with torch.no_grad():
            outputs = self.model(
                input_ids=generated_with_mask,
                attention_mask=torch.ones_like(generated_with_mask),
                use_cache=False, # TODO
            )

        next_token_logits = outputs.logits[:, -self.n_future_tokens:, :]  # 1, n_future_tokens, vocab_size
        _, todo_verify_y0 = self.sample_tokens(next_token_logits, **self.sampling_args)

        self.timers['prefill_end'] = time.time()

        # verify stage
        while generated.size(1) < self.total_generate_len:
            forward_step += 1

            pad_verify_y0 = prepare_verify_map(todo_verify_y0[0]) 
            pad_verify_generated = torch.cat(
                (
                    generated.repeat(self.n_future_tokens, 1),
                    pad_verify_y0.to(device=generated.device, dtype=generated.dtype)
                ), dim=1
            ) # [1 * n_future_tokens, seq_len + n_future_tokens * 2 - 1]
            attention_mask = torch.ones(pad_verify_generated.shape, dtype=generated.dtype, device=generated.device)
            attention_mask[pad_verify_generated == PAD_TOKEN_ID] = 0
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=pad_verify_generated,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            pad_verify_logits = outputs.logits[:, -(self.n_future_tokens*2-1):, :]  # [1 * n_future_tokens, n_future_tokens * 2 - 1]
            _, pad_verify_y1 = self.sample_tokens(pad_verify_logits, **self.sampling_args)
            
            accept_count = 1
            for ix in range(1, self.n_future_tokens):
                if pad_verify_y0[ix, ix] == pad_verify_y1[ix-1, ix-1]:
                    accept_count += 1
                else:
                    break

            new_tokens = todo_verify_y0[0][:accept_count]

            eos_found, new_tokens = self.is_eos_found(new_tokens)
            todo_verify_y0 = pad_verify_y1[accept_count-1, accept_count-1 : accept_count-1 + self.n_future_tokens].unsqueeze(0)
            generated = torch.concat([generated[0], new_tokens.to(generated.device)], dim=-1).unsqueeze(0)

            if self.save_history:
                history_record.append(
                    (self.tokenizer.batch_decode(generated[:, seq_len:]), generated[:, seq_len:].size(1))
                )
            
            if eos_found:
                break
    
        generated_ids = generated[:, seq_len:]
        self.timers['total_end'] = time.time()

        return generated_ids, history_record, forward_step


    def self_speculative_generate_cached(
        self,
        model_inputs: Union[str, torch.Tensor, Dict[str, torch.Tensor]],
    ) -> List[str]:
        batch_size, seq_len = model_inputs.input_ids.shape
        generated  = model_inputs.input_ids.clone()

        self.model.use_cache = True
        self.model.model.decoding_with_ssd_cache = True

        forward_step = 1
        history_record = []
        past_key_values = None
        pad_len = self.n_future_tokens * (self.n_future_tokens - 1) + self.n_future_tokens * (self.n_future_tokens + 1) // 2
                
        def prepare_verify_input_ids(x_out):
            pad_y = torch.full((batch_size, pad_len), self.MASK_TOKEN_ID)        
            cur_ix = 0
            for ix in range(self.n_future_tokens):
                pad_y[0, cur_ix:cur_ix+ix+1] = x_out[:ix+1]
                cur_ix += ix + self.n_future_tokens

            return pad_y
        
        def prepare_verify_pe(start_pe):
            pe = torch.full((batch_size, pad_len), start_pe, dtype=torch.long, device=generated.device)
            cur_ix = 0
            for ix in range(self.n_future_tokens):
                pe[0, cur_ix:cur_ix+ix+1+self.n_future_tokens-1] = torch.arange(
                    start_pe, 
                    start_pe +ix + 1 + self.n_future_tokens - 1, 
                )
                cur_ix += ix + self.n_future_tokens

            return  pe
        
        def get_accept_token(y0, y1):
            reshape_y0, reshape_y1 = [], []
            cur_ix = 0
            for ix in range(self.n_future_tokens):
                reshape_y0.append(y0[cur_ix:cur_ix+ix+1])
                reshape_y1.append(y1[cur_ix:cur_ix+ix+self.n_future_tokens]) # FIXME
                cur_ix += ix + self.n_future_tokens

            accept_count = 1
            for ix in range(1, self.n_future_tokens):
                if reshape_y0[ix][ix] == reshape_y1[ix-1][ix-1]:
                    accept_count += 1
                else:
                    break
            
            new_tokens = reshape_y0[accept_count - 1]
            todo_verify_tokens = reshape_y1[accept_count-1][accept_count-1:]

            return new_tokens, todo_verify_tokens, accept_count

        # prefill stage
        generated_with_mask = torch.cat(
            (
                generated, 
                generated[:, -1].unsqueeze(1),
                torch.full((batch_size, self.n_future_tokens - 1), self.MASK_TOKEN_ID, dtype=generated.dtype, device=generated.device)
            ), dim=1
        ) # [batch_size, seq_len + 1 +  n_future_tokens - 1]

        position_ids = torch.arange(
            past_key_values[0][0].size(2) if past_key_values is not None else 0, 
            generated_with_mask.size(1), 
            device=generated.device
        ).unsqueeze(0) # update pe for kvcache
        position_ids[0, -self.n_future_tokens:] -= 1

        prepare_inputs =  self.model.prepare_inputs_for_generation(
            generated_with_mask,
            past_key_values,
            None,
            use_cache = True, # Note
            position_ids = position_ids
        )

        with torch.no_grad():
            outputs = self.model(**prepare_inputs)

        past_key_values = tuple([
            (item[0][:, :, :generated.shape[1], :], item[1][:, :, :generated.shape[1], :]) 
            for item in outputs.past_key_values
        ])

        next_token_logits = outputs.logits[:, -self.n_future_tokens:, :] # [1, n_future_tokens, vocab_size]
        _, todo_verify_y0 = self.sample_tokens(next_token_logits, **self.sampling_args)

        self.timers['prefill_end'] = time.time()

        # verify stage
        while generated.size(1) < self.total_generate_len:
            forward_step += 1

            pad_varify_y0 = prepare_verify_input_ids(todo_verify_y0[0])
            generated_with_mask = torch.cat(
                (
                    generated, 
                    generated[:, -1].unsqueeze(1),
                    pad_varify_y0.to(device=generated.device, dtype=generated.dtype)
                ), dim=1
            ) # [batch_size, seq_len  + 1 +  pad_len]

            position_ids = torch.arange(past_key_values[0][0].shape[2], generated_with_mask.shape[1])
            position_ids[-pad_len-1] = position_ids[-pad_len-1] - 1
            position_ids[-pad_len:] = prepare_verify_pe(position_ids[-pad_len-1] + 1)
            position_ids = position_ids.unsqueeze(0)

            prepare_inputs =  self.model.prepare_inputs_for_generation(
                generated_with_mask,
                past_key_values,
                None,
                use_cache = True, # Note
                position_ids = position_ids
            )

            with torch.no_grad():
                outputs = self.model(**prepare_inputs)

            past_key_values = tuple([
                (item[0][:, :, :generated.shape[1], :], item[1][:, :, :generated.shape[1], :]) 
                for item in outputs.past_key_values
            ]) # update kvcache

            logits = outputs.logits
            pad_verify_logits = logits[:, -pad_len:, :] # []
            _, pad_verify_y1 = self.sample_tokens(pad_verify_logits, **self.sampling_args)
                        
            new_tokens, todo_verify_tokens, accept_count = get_accept_token(pad_varify_y0[0], pad_verify_y1[0])

            eos_found, new_tokens = self.is_eos_found(new_tokens)
            todo_verify_y0 = todo_verify_tokens.unsqueeze(0)
            generated = torch.concat([generated[0], new_tokens.to(generated.device)], dim=-1).unsqueeze(0)
            
            if self.save_history:
                history_record.append(
                    (self.tokenizer.batch_decode(generated[:, seq_len:]), generated[:, seq_len:].size(1))
                )
            
            if eos_found:
                break
        
        generated_ids = generated[:, seq_len:]
        self.timers['total_end'] = time.time()

        return generated_ids, history_record, forward_step


    def conf_generate_cached(
        self,
        model_inputs: Union[str, torch.Tensor, Dict[str, torch.Tensor]],
        threshold: float = 0.8,
        only_first_token_keep: bool = False
    ):
        batch_size, seq_len = model_inputs.input_ids.shape
        generated = model_inputs.input_ids.clone()

        self.model.use_cache = False

        history_record = []
        forward_step = 0
        past_key_values = None

        while generated.size(1) < self.total_generate_len:
            forward_step += 1

            generated_with_mask = torch.cat(
                (
                    generated, 
                    generated[:, -1].unsqueeze(1),
                    torch.full(
                        (batch_size, self.n_future_tokens-1), 
                        self.MASK_TOKEN_ID, dtype=generated.dtype, device=generated.device
                    )
                ), dim=1
            ) # [batch_size, seq_len + 1 +  n_future_tokens - 1]

            # # update pe for kvcache
            position_ids = torch.arange(
                past_key_values[0][0].size(2) if past_key_values is not None else 0, 
                generated_with_mask.size(1),
                device=generated.device
            ).unsqueeze(0)
            position_ids[0, -self.n_future_tokens:] -= 1 

            prepare_inputs =  self.model.prepare_inputs_for_generation(
                generated_with_mask,
                past_key_values,
                None,
                use_cache = True, # Note
                position_ids = position_ids
            )

            with torch.no_grad():
                outputs = self.model(**prepare_inputs)
            
            past_key_values = tuple([
                (item[0][:, :, :generated.shape[1], :], item[1][:, :, :generated.shape[1], :]) 
                for item in outputs.past_key_values
            ]) # update kvcache

            next_token_logits = outputs.logits[:, -self.n_future_tokens:, :] # [batch_size, n_future_tokens, vocab_size]
            
            confidence, x0 = self.sample_tokens(next_token_logits, **self.sampling_args)
            accept_count = 1 if only_first_token_keep else max(1, min(self.find_longest_sequence(confidence, threshold)[0].item(), self.n_future_tokens))

            new_tokens = x0[0, :accept_count]
            eos_found, new_tokens = self.is_eos_found(new_tokens)

            generated = torch.cat([generated, new_tokens.unsqueeze(0).to(generated.device)], dim=1)
            
            if self.timers['prefill_end'] is None:
                self.timers['prefill_end'] = time.time()

            if self.save_history:
                history_record.append(
                    (self.tokenizer.batch_decode(generated[:, seq_len:]), generated[:, seq_len:].size(1))
                )

            if eos_found:
                break
            
        self.timers['total_end'] = time.time()
        generated_ids = generated[:, seq_len:]

        return generated_ids, history_record, forward_step
        
    def conf_generate(
        self,
        model_inputs: Union[str, torch.Tensor, Dict[str, torch.Tensor]],
        threshold: float = 0.8,
        only_first_token_keep: bool = False
    ):
        batch_size, seq_len = model_inputs.input_ids.shape
        generated = model_inputs.input_ids.clone()

        self.model.use_cache = False

        history_record = []
        forward_step = 0

        while generated.size(1) < self.total_generate_len:
            forward_step += 1
            
            generated_with_mask = torch.cat(
                (
                    generated, 
                    torch.full(
                        (batch_size, self.n_future_tokens-1), 
                        self.MASK_TOKEN_ID, dtype=generated.dtype, device=generated.device
                    )
                ), dim=1
            ) # [batch_size, seq_len +  n_future_tokens - 1]

            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_with_mask,
                    attention_mask=torch.ones_like(generated_with_mask),
                    use_cache=False,
                )

            next_token_logits = outputs.logits[:, -self.n_future_tokens:, :] # [batch_size, seq_len, vocab_size] -> [batch_size, n_future_tokens, vocab_size]

            confidence, x0 = self.sample_tokens(next_token_logits, **self.sampling_args)
            accept_count = 1 if only_first_token_keep else max(1, min(self.find_longest_sequence(confidence, threshold)[0].item(), self.n_future_tokens))

            new_tokens = x0[0, :accept_count]
            eos_found, new_tokens = self.is_eos_found(new_tokens)

            generated = torch.cat([generated, new_tokens.unsqueeze(0).to(generated.device)], dim=1)

            if self.timers['prefill_end'] is None:
                self.timers['prefill_end'] = time.time()

            if self.save_history:
                history_record.append(
                    (self.tokenizer.batch_decode(generated[:, seq_len:]), generated[:, seq_len:].size(1))
                )

            if eos_found:
                break

        self.timers['total_end'] = time.time()
        generated_ids = generated[:, seq_len:]
        
        return generated_ids, history_record, forward_step
    
    
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

    n_future_tokens = model.config.block_size # or setting to a larger D, recommand 8 for SDLM-3B-D4 and 16 for SDLM-3B-D8

    sampling_args = {
        'temperature': 0,
        'top_p': None,
        'top_k': None,
        'entropy_conf': False
    }

    model_generator = SDLMGenerator(
        model,
        tokenizer,
        sampling_args,
        n_future_tokens
    )
    

    prompt = 'Write a Fibonacci function in Python.'
    # prompt = "What fraction of 2 feet is 3 inches? Express your answer as a common fraction.\nPlease reason step by step, and put your final answer within \\boxed{}." # ans \\frac{1}{8}
    
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

    for alg, cached in [
        # ['self_speculative', False],
        ['prob_conf', False],
        # ['prob_conf', True],
        # ['self_speculative', True],
    ]:
        response, history = model_generator.generate(
            model_inputs,
            max_gen_len = 512,
            use_cache=cached,
            alg = alg, #  prob_conf | entropy_conf | self_speculative
            threshold = 0.4,
            save_history = True,
            only_first_token_keep = False,
            n_future_tokens=n_future_tokens
        )

        print(response[0])

        print('\n\n=======histroy')
        for item in history:
            print('cur total token ', item[1])
            print(item[0][0])
            print('--------\n')
