# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import hashlib
import io
import json
import os
import random

import torch
from internvl.train.constants import (IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN)
from internvl.train.dataset import build_transform
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_image(image_url_or_path, _client=None, convert_rgb: bool = True):
    if 's3://' in image_url_or_path:
        image = Image.open(io.BytesIO(_client.get(image_url_or_path)))
    else:
        image = Image.open(image_url_or_path)
    return image.convert('RGB') if convert_rgb else image


def load_json(json_url_or_path, _client=None):
    if 's3://' in json_url_or_path:
        try_times = 0
        bytes = None
        while try_times < 10:
            try:
                bytes = _client.get(json_url_or_path)
                break
            except Exception as e:
                # print(f'Failed to get {json_url_or_path}, retry {try_times}')
                try_times += 1
        return json.load(io.BytesIO(bytes))
    else:
        return json.load(open(json_url_or_path, 'r'))


def load_json_line(line_str, try_times=20):
    _try_times = 0
    while _try_times < try_times:
        try:
            data = json.loads(line_str)
            break
        except Exception as e:
            data = None
            # print(f'Failed to load line, retry {_try_times}')
            _try_times += 1
            line_str = line_str[:-1]
    if data is None:
        raise Exception(f'Failed to get {line_str}')
    return data


def load_jsonl(jsonl_url_or_path, _client=None, decode=True):
    if 's3://' in jsonl_url_or_path:
        try_times = 0
        while try_times < 10:
            try:
                bytes = _client.get(jsonl_url_or_path)
                break
            except Exception as e:
                print(e)
                print(f'Failed to get {jsonl_url_or_path}, retry {try_times}')
                try_times += 1
        lines = io.BytesIO(bytes).readlines()
    else:
        lines = open(jsonl_url_or_path, 'r').readlines()

    if decode:
        data = []
        for line in lines:
            if len(line.strip()) > 2:
                try:
                    sample = load_json_line(line)
                    data.append(sample)
                except Exception as e:
                    raise ValueError(f'Failed to load line: {line}') from e
    else:
        data = lines
    return data


def encode_hash_sha256(txt):
    hash_object = hashlib.sha256(txt.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig


def partition_for_rank(all_rank_item_list: list, rank: int, world_num: int) -> list:
    num_per_rank = len(all_rank_item_list) // world_num
    this_rank_item_list = all_rank_item_list[rank * num_per_rank: (rank + 1) * num_per_rank]
    return this_rank_item_list


class InterleavedDataset(Dataset):

    def __init__(self, meta, tokenizer, tcs_loader, num_image_token=256, image_size=448, is_train=False,
                 pad2square=False, group_by_length=False, normalize_type='imagenet', max_num_images=6,
                 train_num_samples=None, dataset_resampled=True, seed=88):

        self.tokenizer = tokenizer
        self.data_path = meta['annotation']
        self.image_path = meta['root']
        self.max_num_images = max_num_images
        self.train_num_samples = train_num_samples
        self.dataset_resampled = dataset_resampled
        self.tcs_loader = tcs_loader.client
        self.num_image_token = num_image_token
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.group_by_length = group_by_length
        self.normalize_type = normalize_type
        self.sep = '</s>'

        # 0-6143 each 34195 samples
        self.num_samples_each_shard = 34190  # even if the actual num is more
        self._length = self.num_samples_each_shard * 6144

        self.random = random.Random(seed)
        shard_order = list(range(6144))
        self.world_size = torch.distributed.get_world_size()
        current_rank = torch.distributed.get_rank()
        shard_order = partition_for_rank(shard_order, rank=current_rank, world_num=self.world_size)

        if self.dataset_resampled:
            self.random.shuffle(shard_order)

        self.raw_data = []
        for i in range(len(shard_order)):
            shard_name = f'data0417_shuffled_shard_{shard_order[i]}.jsonl'
            shard_data = load_jsonl(os.path.join(self.data_path, shard_name), self.tcs_loader, decode=False)
            print(f'{shard_name} has {len(shard_data)} samples')
            self.raw_data.extend(shard_data)
        print('raw_data has', len(self.raw_data), 'samples')
        self.local_length = len(self.raw_data)
        self.random.shuffle(self.raw_data)

    def load_ann_file(self, file_path):
        if file_path.endswith('.json'):
            return load_json(file_path, self.tcs_loader)
        elif file_path.endswith('.jsonl'):
            return load_jsonl(file_path, self.tcs_loader)
        else:
            raise NotImplementedError(f'Unsupported annotation file format: {file_path}')

    def __len__(self):
        return self._length

    def load_data(self, index):
        index = index % self.local_length
        data = json.loads(self.raw_data[index])
        return data

    def get_img_filename(self, web_url):
        return self.encode_hash_sha256(web_url)

    @staticmethod
    def encode_hash_sha256(web_url):
        hash_object = hashlib.sha256(web_url.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig

    def load_image(self, image_path_or_url):
        try:
            if 's3://' in self.image_path:
                # load from aws ceph
                return Image.open(io.BytesIO(self.tcs_loader.get(image_path_or_url))).convert('RGB')
            else:
                # load from local (or s3mount node)
                return Image.open(image_path_or_url).convert('RGB')
        except Exception as err:
            print(f'Error loading image: {image_path_or_url}: {err}')
            return None

    def parse_sample(self, sample):
        images = sample['images']
        texts = sample['texts']
        metadata = sample.get(
            'metadata',
            [
                {'filename': self.encode_hash_sha256(web_url)}
                if web_url is not None else None
                for web_url in images
            ]
        )
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        assert isinstance(metadata, list), metadata
        valid_image = sample.get('valid_image', [True] * sum(img is not None for img in images))
        assert len(images) == len(texts)
        assert sum(img is not None for img in images) == sum(txt is None for txt in texts) == len(valid_image), (
            sum(img is not None for img in images), sum(txt in ['<image>', None] for txt in texts), len(valid_image),
            sample)
        for _img, _imgmeta in zip(images, metadata):
            assert (_img is None) == (_imgmeta is None), sample
        return images, texts, metadata, valid_image

    def preprocess_image(self, images):
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        images = [transform(image) for image in images]
        images = torch.stack(images, dim=0)
        return images

    def pure_text_get_item(self, texts):
        text = '\n\n'.join([_ for _ in texts if _]) + self.sep
        tokenized = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            padding=True,
            return_tensors='pt',
        )
        input_ids = tokenized['input_ids']
        images = [Image.new('RGB', (224, 224), (255, 255, 255))]
        pixel_values = self.preprocess_image(images)
        num_patches = pixel_values.size(0)
        labels = input_ids.clone()
        ret = dict(
            input_ids=input_ids[0],
            labels=labels[0],
            attention_mask=input_ids[0].ne(self.tokenizer.pad_token_id),
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def multimodal_get_item(self, images, texts, valid_image, ):
        image_idx = 0
        for i in range(len(texts)):
            if texts[i] is None:
                if valid_image[image_idx]:
                    texts[i] = '<image>'
                image_idx += 1
        text = '\n\n'.join([_ for _ in texts if _]) + self.sep
        # format cleanup
        text = text.replace('<image>\n\n', '<image>').replace('\n\n<image>', '<image>')
        image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * self.num_image_token}{IMG_END_TOKEN}'
        text = text.replace('<image>', image_tokens, len(images))
        tokenized = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            padding=True,
            return_tensors='pt',
        )
        input_ids = tokenized['input_ids']
        image_start_token_id = self.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        image_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        # count IMG_END_TOKEN
        num_image_end_tokens = input_ids.eq(image_end_token_id).sum().item()
        if num_image_end_tokens != len(images):
            text = text.replace(image_tokens, '<image>', len(images))
            images = images[:num_image_end_tokens]
            # assert len(images) > 0, 'The number of images should be greater than 0.'
            text = text.replace('<image>', image_tokens, len(images))
            text = text.replace('<image>', '')
            tokenized = self.tokenizer(
                text,
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                padding=True,
                return_tensors='pt',
            )
            input_ids = tokenized['input_ids']

        pixel_values = self.preprocess_image(images)
        num_patches = pixel_values.size(0)
        labels = input_ids.clone()
        assert (labels == image_context_token_id).sum() == self.num_image_token * len(
            images), 'image tokens are truncated'
        labels[labels == image_start_token_id] = -100
        labels[labels == image_end_token_id] = -100
        labels[labels == image_context_token_id] = -100
        ret = dict(
            input_ids=input_ids[0],
            labels=labels[0],
            attention_mask=input_ids[0].ne(self.tokenizer.pad_token_id),
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def getitem(self, index):
        # dict_keys(['general_metadata', 'images', 'texts', 'metadata', 'doc_loc'])
        sample = self.load_data(index)
        assert sample is not None

        # parse sample and check
        images, texts, metadata, valid_image = self.parse_sample(sample)
        # get valid images
        images = [os.path.join(self.image_path, self.get_img_filename(img)) for img, _ in
                  zip(images, metadata) if img is not None]

        loaded_images = []
        valid_count = 0
        for idx, (img, valid) in enumerate(zip(images, valid_image)):
            if valid:
                if valid_count >= self.max_num_images:
                    valid_image[idx] = False
                else:
                    _image = self.load_image(img)
                    if _image is not None:
                        loaded_images.append(_image)
                        valid_count += 1
                    else:
                        valid_image[idx] = False

        if len(loaded_images) > 0:
            ret = self.multimodal_get_item(loaded_images, texts, valid_image)
        else:
            ret = self.pure_text_get_item(texts)
        return ret

    def __getitem__(self, index):
        while True:
            try:
                index = index % self._length
                item = self.getitem(index)
                break
            except Exception as err:
                index = index + 1
                print(f'Try to load index {index} again, due to {err}')
        return item


if __name__ == '__main__':
    from tqdm import tqdm

    model_path = '/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat_dev/release/InternVL-Chat-V1-5'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.model_max_length = 4096

    metas = {
        'lmm_interleaved_data0417_shuffled': {
            'root': 'wwhnew_pssd:s3://mllm-cc/raw-images/',
            'annotation': 'langchao:s3://liqingyun/projects/lmm_interleaved/data0417_shuffled/',
            'data_augment': True,
            'repeat_time': 1,
            'length': 210063360
        },
    }
    from internvl.train.dataset import TCSLoader

    tcs_loader = TCSLoader('~/petreloss.conf')
    dataset = InterleavedDataset(meta=metas['lmm_interleaved_data0417_shuffled'],
                                 tokenizer=tokenizer,
                                 tcs_loader=tcs_loader)
    for i in tqdm(range(1000)):
        item = dataset.__getitem__(i)
    print(f'length: {len(dataset)}')
