# Evaluation

We use OpenCompass for model evaluation. Currently, evaluation parameters need to be manually configured in `eval/with_opencompass/oc_sdlm.py`.

## SetUp

1. putting generator `eval/with_opencompass/sdlm` and ``eval/with_opencompass/oc_sdlm.py` to `opencompass/models`

|-- sdlm
|   |-- __init__.py
|   `-- sdlm_generate.py
|-- oc_sdlm.py

2. for model card, refer to `eval/with_opencompass/SDLM_3B_D4.py`

## Configuration Settings

- max_seq_length: 2048
- temperature: 0
- top_k: None
- top_p: None
- n_future_tokens: 4
- only_first_token_keep: False
- **alg**: 'prob_conf'
- **threshold**: 0.98


## Evaluation Benchmarks
The following benchmarks are used for comprehensive evaluation:

### Mathematical Reasoning:

- math_500_gen
- gsm8k_0shot_v2_gen_6e39a4
- gpqa_gen_4baadb

### Code Generation:

- humaneval_gen
- humaneval_plus_gen
- sanitized_mbpp_mdblock_gen_a447ff
- mbpp_plus_gen (378 instances)

### Instruction Following:

- IFEval_gen

### General Knowledge & Reasoning:

- ARC_e_gen
- ARC_c_gen
- hellaswag_gen
- winogrande_gen
- mmlu_gen_4d595a

## Environment

<details>
  <summary>Environment</summary>

The evaluation environment uses:
```
{'CUDA available': True,
'CUDA_HOME': '/mnt/petrelfs/share/cuda-12.1',
'GCC': 'gcc (GCC) 9.4.0',
'GPU 0': 'NVIDIA A800-SXM4-80GB',
'MMEngine': '0.10.6',
'MUSA available': False,
'NVCC': 'Cuda compilation tools, release 12.1, V12.1.66',
'OpenCV': '4.11.0',
'PyTorch': '2.1.0+cu121',
'PyTorch compiling details': 'PyTorch built with:\n'
' - GCC 9.3\n'
' - C++ Version: 201703\n'
' - Intel(R) oneAPI Math Kernel Library Version '
'2022.2-Product Build 20220804 for Intel(R) 64 '
'architecture applications\n'
' - Intel(R) MKL-DNN v3.1.1 (Git Hash '
'64f6bcbcbab628e96f33a62c3e975f8535a7bde4)\n'
' - OpenMP 201511 (a.k.a. OpenMP 4.5)\n'
' - LAPACK is enabled (usually provided by '
'MKL)\n'
' - NNPACK is enabled\n'
' - CPU capability usage: AVX512\n'
' - CUDA Runtime 12.1\n'
' - NVCC architecture flags: '
'-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90\n'
' - CuDNN 8.9.2\n'
' - Magma 2.6.1\n'
' - Build settings: BLAS_INFO=mkl, '
'BUILD_TYPE=Release, CUDA_VERSION=12.1, '
'CUDNN_VERSION=8.9.2, '
'CXX_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/c++, '
'CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=0 '
'-fabi-version=11 -fvisibility-inlines-hidden '
'-DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO '
'-DLIBKINETO_NOROCTRACER -DUSE_FBGEMM '
'-DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK '
'-DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE '
'-O2 -fPIC -Wall -Wextra -Werror=return-type '
'-Werror=non-virtual-dtor -Werror=bool-operation '
'-Wnarrowing -Wno-missing-field-initializers '
'-Wno-type-limits -Wno-array-bounds '
'-Wno-unknown-pragmas -Wno-unused-parameter '
'-Wno-unused-function -Wno-unused-result '
'-Wno-strict-overflow -Wno-strict-aliasing '
'-Wno-stringop-overflow -Wno-psabi '
'-Wno-error=pedantic -Wno-error=old-style-cast '
'-Wno-invalid-partial-specialization '
'-Wno-unused-private-field '
'-Wno-aligned-allocation-unavailable '
'-Wno-missing-braces -fdiagnostics-color=always '
'-faligned-new -Wno-unused-but-set-variable '
'-Wno-maybe-uninitialized -fno-math-errno '
'-fno-trapping-math -Werror=format '
'-Werror=cast-function-type '
'-Wno-stringop-overflow, LAPACK_INFO=mkl, '
'PERF_WITH_AVX=1, PERF_WITH_AVX2=1, '
'PERF_WITH_AVX512=1, '
'TORCH_DISABLE_GPU_ASSERTS=ON, '
'TORCH_VERSION=2.1.0, USE_CUDA=ON, USE_CUDNN=ON, '
'USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, '
'USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, '
'USE_MPI=OFF, USE_NCCL=1, USE_NNPACK=ON, '
'USE_OPENMP=ON, USE_ROCM=OFF, \n',
'Python': '3.10.16 | packaged by conda-forge | (main, Dec 5 2024, 14:16:10) '
'[GCC 13.3.0]',
'TorchVision': '0.16.0+cu121',
'lmdeploy': "not installed:No module named 'lmdeploy'",
'numpy_random_seed': 2147483648,
'opencompass': '0.4.2+3f50b1d',
'sys.platform': 'linux',
'transformers': '4.37.2'}
```

</details>
