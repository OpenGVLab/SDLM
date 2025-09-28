from opencompass.models import SDLM_Qwen2_5

models = [
    dict(
        type=SDLM_Qwen2_5,
        abbr='SDLMQwen2_5',
        path='SDLM_3B_D4',
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]