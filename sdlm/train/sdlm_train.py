# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import sys
import math
import random
import logging
import warnings
import traceback
import numpy as np
from functools import partial
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

try:
    import orjson as json
except:
    import json

import torch
import transformers
import torch.distributed as dist
from sdlm.dist_utils import init_dist
from sdlm.patch import (
    concat_pad_data_collator_pure_text,
    replace_train_dataloader, 
    replace_train_sampler
)
from sdlm.train.dataset import (
    ConcatDataset, 
    WeightedConcatDataset
)

from torch.utils.data import Dataset
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    HfArgumentParser, 
    Trainer,
    TrainingArguments,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format, 
    set_verbosity
)


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from sdlm.train.trainer_log_loss import CustomTrainerForLogloss, TensorBoardLoggingCallback
from sdlm.preprocess.preprocess_sdlm import preprocess_mask_block


import torch.nn.functional as F
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

IGNORE_INDEX = -100
TEXT_MASK_TOKEN = '<text_mask>'


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    block_size: int = field(
        default=4,
        metadata={'help': 'block size of mask token.'},
    )
    causal_attn: bool = field(
        default=False,
        metadata={'help': 'causal attention or not. default to False.'},
    )
    attn_implementation: Optional[str] = field(
        default='eager',
        metadata={'help': ''},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        ds_name,
        is_train=True,
        repeat_time=1,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
        block_size=4,
        text_mask_token_id=151666
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name

        self.is_train = is_train

        # hyperparameters for distributed training
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        # TODO: quick resume
        self._state_dict = {}

        self.block_size = block_size
        self.text_mask_token_id = text_mask_token_id

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time <= 1:
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        return partial(
            preprocess_mask_block,
            block_size=self.block_size,
            mask_token_id=self.text_mask_token_id,
            expected_mask_repeat_times=3 
        )


    def pure_text_get_item(self, data_item):
        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(
            self.template_name, 
            [deepcopy(data_item['conversations'])],
            self.tokenizer, 
            ds_name=self.ds_name
        )

        assert ret['position_ids'].shape == ret['attention_mask'].shape == ret['input_ids'].shape == ret['labels'].shape

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=ret['position_ids'][0]
        )
        return ret

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, data_item, flush=True)
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            yield self[i]


def build_datasets(
    data_args,
    tokenizer,
    block_size=4,
    text_mask_token_id=151666
):
    datasets = []
    lengths = []
    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        dataset = LazySupervisedDataset(
            data_args.conv_style, 
            ds_collections[ds_name],
            tokenizer,
            ds_name=ds_name,
            is_train=ds_collections[ds_name]['data_augment'],
            repeat_time=repeat_time,
            data_rank=data_rank,
            data_world_size=data_world_size,
            random_seed=ds_idx,
            block_size=block_size,
            text_mask_token_id=text_mask_token_id
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def main():
    # Apply necessary patches for the transformers library
    replace_train_sampler()
    replace_train_dataloader()

    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    logger.info(f'max seq length: {data_args.max_seq_length}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token=True,  # for im_end
        trust_remote_code=True,
        use_fast=model_args.use_fast_tokenizer
    )
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length

    token_list = [TEXT_MASK_TOKEN] 
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)


    text_mask_token_id = tokenizer.convert_tokens_to_ids(TEXT_MASK_TOKEN)
    print(f'{TEXT_MASK_TOKEN=} {text_mask_token_id=}')

    if model_args.model_name_or_path is not None:
        from sdlm.model.sdlm_qwen2_5 import Qwen2ForCausalLM
        assert model_args.attn_implementation in ['flash_attention_2', 'eager', 'sdpa']
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        
        model = Qwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=model_args.attn_implementation,  # TODO flash_attention_2 or eager or sdpa
        )

        model.to(training_args.device)
        model.config.use_cache = False
        model.training = True
        model.text_mask_token_id = text_mask_token_id

        model.model.block_size = int(model_args.block_size)
        model.model.causal_attn = model_args.causal_attn
        model.model.training = True # FIXME need to setting before training.
        
        config.block_size = int(model_args.block_size)
        config.causal_attn =  model_args.causal_attn
        config.text_mask_token_id = text_mask_token_id
        config.text_mask_token = TEXT_MASK_TOKEN

        logger.info(f'Loading {model_args.model_name_or_path}...')
    else:
        raise ValueError

    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        model.config.vocab_size = len(tokenizer)

    if model_args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    print(model, model.config)

    print(f'{model.config.bos_token_id=} {model.training=}')
    print(f'{model.model.causal_attn=}, {model.model.block_size=} {model.model.training=}')

    train_dataset = build_datasets(
        data_args, 
        tokenizer, 
        block_size=model_args.block_size, 
        text_mask_token_id=text_mask_token_id
    )

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)


    collator = partial(
        concat_pad_data_collator_pure_text,
        max_item_length=None,
        pad_id=model.config.bos_token_id
    )

    trainer = CustomTrainerForLogloss(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[TensorBoardLoggingCallback()]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()

        config.save_pretrained(training_args.output_dir)


if __name__ == '__main__':
    main()
