import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    configure_pad_token,
    get_dtype,
    handle_stop_sequences,
    pad_and_concat,
    stop_sequences_criteria,
)

from transformers import AutoTokenizer, AutoModel
from sdlm.model.sdlm_qwen2_5.modeling_qwen2 import Qwen2ForCausalLM
from sdlm.model.sdlm_qwen2_5.generate_utils import SDLM_generate

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("qwen_mask_block")
class SDLM_Qwen2_5(LM):
    def __init__(
        self,
        model_path='',
        mask_id=151666,
        max_length=6144,
        batch_size=1,
        gen_length=1024,
        temperature=0,
        top_k=None,
        top_p=None,
        threshold=0.98,
        n_future_tokens=4,
        only_first_token_keep=False,
        use_cache=True,
        alg='maskgit_plus',
        device="cuda",
        **kwargs,
    ):
        '''
        Args:
            model_path: model path.
            mask_id: The token id of [MASK] is 151666.
            max_length: the max sequence length.
            batch_size: mini batch size.
        '''
        super().__init__()

        try:
            accelerator = accelerate.Accelerator()
            if accelerator.num_processes > 1:
                self.accelerator = accelerator
            else:
                self.accelerator = None
        except:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        model_kwargs['attn_implementation'] = 'eager'

        self.model = Qwen2ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.batch_size = int(batch_size)
        self.max_length = max_length

        self.gen_length = gen_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.threshold = threshold
        self.n_future_tokens = n_future_tokens
        self.only_first_token_keep = only_first_token_keep
        self.use_cache = use_cache
        self.alg = alg
        print('init...')

        # print(f'{self.model.model.block_size=}')
        # print(f'{self.model=}')
        # print(f'Model initialized on {self.device}, rank {self._rank}/{self._world_size}')

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        logits = self.model(batch).logits
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        raise NotImplementedError

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": e["question"]}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return {
                "prompt": prompt,
                "question_text": e["question"],
                "until": e["until"],
            }
        
        
        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)

        out = []
        for elem in tqdm(ds, desc="Generating..."):
            prompt = elem["prompt"]
            stop_tokens = elem["until"]

            model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
            resp = SDLM_generate(
                self.model,
                self.tokenizer,
                model_inputs,
                max_gen_len = self.gen_length,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                threshold=self.threshold,
                n_future_tokens=self.n_future_tokens,
                only_first_token_keep =self.only_first_token_keep,
                use_cache = self.use_cache,
                alg=self.alg
            )[0]
            
            for stop_seq in stop_tokens:
                if stop_seq in resp:
                    resp = resp.split(stop_seq)[0].strip()

            # print('response\n', resp, '\n\n')
            out.append(resp)

            if self.accelerator:
                self.accelerator.wait_for_everyone()

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
    