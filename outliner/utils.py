import datasets
import torch
from transformer_lens import HookedTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
base_model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
tokenizer = base_model.tokenizer

def get_owt_dataset(min_len=1024):
    dataset : IterableDataset = datasets.load_dataset(path="openwebtext", streaming=True)["train"] # type: ignore
    dataset = dataset.shuffle(buffer_size=10_000, seed=0)

    def add_tok_columns(dataset_batch):
        dataset_batch["toks"] = base_model.to_tokens(dataset_batch["text"])
        dataset_batch["seqlen"] = (dataset_batch["toks"] != tokenizer.pad_token_id).sum(-1) + 1
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
