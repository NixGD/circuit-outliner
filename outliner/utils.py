from functools import cache

import datasets
import torch
from transformer_lens import HookedTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@cache
def get_base_model(model_name: str = "gpt2-small"):
    """Cached wrapper to not duplicate weights in memory"""
    return HookedTransformer.from_pretrained(model_name, device=DEVICE)


def get_owt_dataset(min_len=1024):
    """
    Assumes GPT2 style tokenizer
    """
    base_model = get_base_model() 
    dataset : IterableDataset = datasets.load_dataset(path="openwebtext", streaming=True)["train"] # type: ignore
    dataset = dataset.shuffle(buffer_size=10_000, seed=0)

    def add_tok_columns(dataset_batch):
        dataset_batch["toks"] = base_model.to_tokens(dataset_batch["text"])
        dataset_batch["seqlen"] = (dataset_batch["toks"] != base_model.tokenizer.pad_token_id).sum(-1) + 1
        return dataset_batch

    dataset = dataset.map(add_tok_columns, batched=True, batch_size=10)
    dataset = dataset.filter(lambda x: x["seqlen"] >= min_len)
    return dataset


class ReverseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return (-grad_output)

def kl_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Expects both a and b to be logit tensors of shape [batch, seqlen, vocab].
    Sums over last dimension, then averages.
    """
    return torch.nn.functional.kl_div(a.log_softmax(-1), b.log_softmax(-1), log_target=True, reduction='none').sum(-1).mean()
