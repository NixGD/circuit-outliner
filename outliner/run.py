# %%
import dataclasses
import itertools
from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn
from tqdm import trange

import wandb
from mixer import MixerHparams
from mixing_transformer import DirectOutMixingTransformer, MixingTransformer, MixingTransformerHparams
from utils import DEVICE, base_model, get_owt_dataset, tokenizer

# %%
# mix_model = MixingTransformer(base_model)

dataset = get_owt_dataset()


# %% [markdown]

"""
Let's evaluate the loss of the mixed model for different values of p.
At p=0 we are just running the model on the alt-input, but evaluating it's output against the next token from the ref-input. This results in a high loss, as the model is confidently wrong.

As p increases the loss should decrease, as more information from the reference prompt passes through.

At p=1 we are just running the model on the reference input, so the loss should be the same as the base model (about 3 for GPT2-sm on open web text).
"""

# %%


def get_loss(model, batch_iter) -> torch.Tensor:
    ref_toks = torch.stack(next(batch_iter)["toks"])
    alt_toks = torch.stack(next(batch_iter)["toks"])
    out = model.forward(ref_toks, alt_toks)

    loss = torch.nn.CrossEntropyLoss(reduction="mean")(out[:, :-1].flatten(0, 1), ref_toks[:, 1:].flatten())
    return loss


def eval_p(p: float | torch.Tensor) -> None:
    hparams = MixingTransformerHparams(mixer_hparams=MixerHparams(init_p=p))
    model = DirectOutMixingTransformer(base_model, hparams=hparams)

    batch_iter = dataset.iter(batch_size=4)
    losses = []
    with torch.no_grad():
        losses = [get_loss(model, batch_iter).item() for _ in range(2)]

    avg_loss = sum(losses) / len(losses)
    print(f"p={p:.2f} | loss={avg_loss:.2f}")


[eval_p(p) for p in torch.arange(0, 1.01, 0.1)]

# %% [markdown]

"""
Some performance notes:
```
> %timeit torch.stack(next(batch_iter)["toks"])
50.9 ms

For batch_size=4:
> %timeit base_model.forward(toks, return_type="loss")
120 ms
> %timeit mixed_model.forward(ref_toks, alt_toks)
231 ms
```
"""
# %%


@dataclass
class TrainHparams:
    transformer_hparams: MixingTransformerHparams = MixingTransformerHparams()
    batch_size: int = 4
    steps: int = 500
    lr: float = 1e-4
    lr_gamma: float = 0.99
    direct_out: bool = False


def log_wandb(step, snapshot, **kwargs):
    to_log = kwargs
    to_log["heads/p/mean"] = snapshot.heads[0].mean().item()
    wandb.log(to_log)


def train_loop(hparams: TrainHparams):
    run: Run = wandb.init(project="outliner", config=dataclasses.asdict(hparams))  # type: ignore

    if hparams.direct_out:
        model = DirectOutMixingTransformer(base_model, hparams=hparams.transformer_hparams)
    else:
        model = MixingTransformer(base_model, hparams=hparams.transformer_hparams)
    opt = torch.optim.Adam(model.mixer_parameters(), lr=hparams.lr)
    schedule = torch.optim.lr_scheduler.ExponentialLR(opt, hparams.lr_gamma)
    batch_iter = dataset.iter(hparams.batch_size)

    for step in trange(hparams.steps):
        loss = get_loss(model, batch_iter)
        opt.zero_grad()
        loss.backward()
        opt.step()
        schedule.step()

        log_wandb(step, model.parameter_snapshot(), loss=loss.item(), lr=schedule.get_last_lr())

    run.finish()
    return model


# %% [markdown]
"""
A basic sanity check: if we have no regularization penalty or weight decay, we should see p increase to ~1.
"""

hparams = TrainHparams(
    transformer_hparams=MixingTransformerHparams(mixer_hparams=MixerHparams(init_p=0.5)),
    steps=200,
    lr=0.03,
    direct_out=True,
)


train_loop(hparams)


# %%
##### Misc testing


# batch = ["The quick brown fox jumped over the lazy dog", "The dog didn't care much. It was used to it."]
# toks = base_model.to_tokens(batch)

# out = mix_model.forward(toks[[0]], toks[[1]])

# %%

# print("Mixed (ref) top predicitons:")
# for i in range(10):
#     top_tok_id = out[0, i].argmax(-1)
#     print(
#         base_model.to_str_tokens(batch[0])[i],
#         repr(base_model.to_string(top_tok_id.item())),
#         out[0, i, top_tok_id].item(),
#         sep="\t| ",
#     )
