CKPT_PATH='SDLM_3B_D4_prob82_all'

srun -N 1 \
    --gres=gpu:8 \
    --cpus-per-task 80 \
    --quotatype=auto \
    python run.py \
        --datasets humaneval_gen gsm8k_0shot_v2_gen_6e39a4 gpqa_gen_4baadb sanitized_mbpp_mdblock_gen_a447ff math_500_gen IFEval_gen mbpp_plus_gen humaneval_plus_gen  \
        --models SDLM_3B_D4  \
        --max-num-workers 8 \
    2>&1 | tee -a "outputs/record/SDLM/${CKPTm_PATH}.txt"

# ARC_c_gen ARC_c_cot_gen_926652 ARC_e_gen hellaswag_gen winogrande_gen mmlu_gen_4d595a
