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
from sdlm_inference import SDLM_generate

if __name__ == "__main__":
    ckpt_hf = './shell/playground/train_states/SDLM_3B_D4/'

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_hf, 
        attn_implementation="eager",
        trust_remote_code=True
    ).to(dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_hf)

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
        n_future_tokens = 4,
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
