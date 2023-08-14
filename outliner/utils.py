import json
import os
from functools import cache
from typing import Callable, Optional

import datasets
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_FOLDER = "../images/"


@cache
def get_base_model(model_name: str = "gpt2"):
    """Cached wrapper to not duplicate weights in memory"""
    return HookedTransformer.from_pretrained(model_name, device=DEVICE)


def get_dataset(
    dataset_name: str = "openwebtext",
    dataset_subset: Optional[str] = None,
    model: str = "gpt2",
    min_len=1024,
):
    dataset: IterableDataset = datasets.load_dataset(path=dataset_name, name=dataset_subset, streaming=True)["train"]  # type: ignore
    dataset = dataset.shuffle(buffer_size=10_000, seed=0)

    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token

    def add_tok_columns(dataset_batch):
        tok_out = tokenizer(
            dataset_batch["text"],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding="max_length",
        ).to(DEVICE)
        dataset_batch["toks"] = tok_out["input_ids"]
        dataset_batch["seqlen"] = (tok_out["input_ids"] != 50256).sum(-1) + 1 # includes start tok
        return dataset_batch

    dataset = dataset.map(lambda x: {"text": "<|endoftext|>" + x["text"]})
    dataset = dataset.map(add_tok_columns, batched=True, batch_size=10)
    dataset = dataset.filter(lambda x: x["seqlen"] >= min_len)
    return dataset


def next_toks(batch_iter):
    return torch.stack(next(batch_iter)["toks"])


class ReverseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


def kl_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Expects both a and b to be logit tensors of shape [batch, seqlen, vocab].
    Sums over last dimension, then averages.
    """
    return (
        torch.nn.functional.kl_div(
            a.log_softmax(-1), b.log_softmax(-1), log_target=True, reduction="none"
        )
        .sum(-1)
        .mean()
    )
