import torch
from transformers import AutoTokenizer
from sdlm.model.sdlm_qwen2_5 import Qwen2ForCausalLM
from sdlm.model.sdlm_qwen2_5.generate_utils import SDLM_generate


if __name__ == "__main__":
    ckpt_name = './shell/playground/train_states/SDLM_3B_D8'

    model = Qwen2ForCausalLM.from_pretrained(
        ckpt_name,
        attn_implementation="eager"
    ).to(
        # device='cuda', 
        dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt_name)


    prompt = "My question is: A cake has 8 slices and each slice contains 347 calories. A pan of brownies has 6 brownies and each slice contains 375 calories. How many more calories does the cake have? Your thoughts:" # ans 526
    prompt = 'Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?'
    prompt = "In a convex quadrilateral, the measure of the largest angle is twice the measure of the smallest angle, and the other two angles are both right angles. How many degrees are in the largest angle?\nPlease reason step by step, and put your final answer within \\boxed{}.\n"  # ans 120, llada's case study
    prompt = "In the land of Ink, the money system is unique. One Trinket is equal to 4 Blinkets, and 3 Blinkets are equal to 7 Drinkets. In Trinkets, what is the value of 56 Drinkets?\nPlease reason step by step, and put your final answer within \\boxed{}.\n"  # ans 6
    prompt = "What fraction of 2 feet is 3 inches? Express your answer as a common fraction.\nPlease reason step by step, and put your final answer within \\boxed{}."  # ans \\frac{1}{8}
    # prompt = "Four distinct circles are drawn in a plane. What is the maximum number of points where at least two of the circles intersect?\nPlease reason step by step, and put your final answer within \\boxed{}." # ans 12

    # prompt = '9 * 3 + 5 = ?'
    # prompt = 'introduce large language model.'
    # prompt = 'Who are you.'
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
        max_gen_len=128,
        temperature=0,
        threshold=0.5,
        n_future_tokens=8,
        alg='prob_conf', #  prob_conf | entropy_conf | self_speculative
        save_history=True,
        use_cache=True
    )

    print(f'{ckpt_name=}')
    print(text)

    print('======')
    print(response[0])

    print('=======histroy')
    for item in history:
        print('cur total token ', item[1])
        print(item[0][0])
        print('--------')


    # print(history)

    # generated_ids = model.generate(
    #     **model_inputs,
    #     max_new_tokens=512
    # )
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]

    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(response)

