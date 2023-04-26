#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on Fairseq code
"""

import logging
from typing import Tuple
import torch
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta.model import RobertaModel as FaiseqRobertaModel
from fairseq.optim.adam import FairseqAdam
from torch import Tensor as T
from torch import nn
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_bert import BertTokenizer

from dpr.models.hf_models import get_roberta_tensorizer
from .biencoder import BiEncoder
from .reader import Reader
from dpr.utils.data_utils import Tensorizer

logger = logging.getLogger(__name__)


def get_roberta_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return RobertaTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )

def get_roberta_biencoder_components(args, inference_only: bool = False, **kwargs):
    question_encoder = RobertaEncoder.from_pretrained(args.pretrained_file)
    ctx_encoder = RobertaEncoder.from_pretrained(args.pretrained_file)
    biencoder = BiEncoder(question_encoder, ctx_encoder)
    optimizer = get_fairseq_adamw_optimizer(biencoder, args) if not inference_only else None

    tensorizer = get_roberta_tensorizer(args)

    return tensorizer, biencoder, optimizer

def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer

class BertTensorizer(Tensorizer):
    def __init__(
        self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self, text: str, title: str = None, add_special_tokens: bool = True
    ):
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length,
                pad_to_max_length=False,
                truncation=True,
            )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (
                seq_len - len(token_ids)
            )
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(
            tokenizer, max_length, pad_to_max=pad_to_max
        )

def get_fairseq_adamw_optimizer(model: nn.Module, args):
    setattr(args, 'lr', [args.learning_rate])
    return FairseqAdam(args, model.parameters()).optimizer

def get_roberta_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_roberta_tokenizer(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case
        )
    return RobertaTensorizer(tokenizer, args.sequence_length)

def get_roberta_reader_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, "dropout") else 0.0
    
    
    encoder = RobertaEncoder.from_pretrained(args.pretrained_file)
    hidden_size = 728

    #ctx_encoder = RobertaEncoder.from_pretrained(args.pretrained_file)
    reader = BiEncoder(encoder, hidden_size)
    optimizer = get_fairseq_adamw_optimizer(reader, args) if not inference_only else None

    tensorizer = get_roberta_tensorizer(args)
    
    #encoder = RobertaEncoder.init_encoder(
    #    args.pretrained_model_cfg, projection_dim=args.projection_dim, dropout=dropout
    #)
    #encoder = RobertaEncoder.from_pretrained(args.pretrained_file)

    #hidden_size = encoder.config.hidden_size
    #reader = Reader(encoder, hidden_size)

    # optimizer = (
    #     get_optimizer(
    #         reader,
    #         learning_rate=args.learning_rate,
    #         adam_eps=args.adam_eps,
    #         weight_decay=args.weight_decay,
    #     )
    #     if not inference_only
    #     else None
    # )

    #optimizer = get_fairseq_adamw_optimizer(reader,args)

    #tensorizer = get_roberta_tensorizer(args)
    return tensorizer, reader, optimizer


class RobertaEncoder(nn.Module):

    def __init__(self, fairseq_roberta_hub: RobertaHubInterface):
        super(RobertaEncoder, self).__init__()
        self.fairseq_roberta = fairseq_roberta_hub

    @classmethod
    def from_pretrained(cls, pretrained_dir_path: str):
        model = FaiseqRobertaModel.from_pretrained(pretrained_dir_path)
        return cls(model)

    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T) -> Tuple[T, ...]:
        roberta_out = self.fairseq_roberta.extract_features(input_ids)
        cls_out = roberta_out[:, 0, :]
        return roberta_out, cls_out, None

    def get_out_size(self):
        raise NotImplementedError
