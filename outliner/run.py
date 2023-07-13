# %%
import dataclasses

import matplotlib.pyplot as plt
import torch
from tqdm import trange

import wandb
from mixer import MixerHparams
from mixing_transformer import (DirectOutMixingTransformer,
                                IndirectMixingTransformer,
                                MixingTransformerHparams)
from utils import base_model, get_owt_dataset

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
    with torch.no_grad():
        loss = get_loss(model, batch_iter).item()
    print(f"p={p:.2f} | loss={loss:.2f}")


[eval_p(p) for p in torch.arange(0, 1.01, 0.2)]

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


@dataclasses.dataclass
class TrainHparams:
    transformer_hparams: MixingTransformerHparams = MixingTransformerHparams()
    batch_size: int = 4
    steps: int = 500
    lr: float = 1e-4
    lr_gamma: float = 0.99
    direct_out: bool = False
    reg_coeff: float = 1
    wandb_enable: bool = True


test_hypers = TrainHparams(steps=10, wandb_enable=False)


def log_wandb(step, snapshot, **kwargs):
    to_log = kwargs
    to_log["heads/p/mean"] = snapshot.heads[0].mean().item()
    wandb.log(to_log)


def train_loop(hparams: TrainHparams):
    if hparams.wandb_enable:
        run: Run = wandb.init(project="outliner", config=dataclasses.asdict(hparams))  # type: ignore

    if hparams.direct_out:
        model = DirectOutMixingTransformer(base_model, hparams=hparams.transformer_hparams)
    else:
        model = IndirectMixingTransformer(base_model, hparams=hparams.transformer_hparams)
    opt = torch.optim.Adam(model.mixer_parameters(), lr=hparams.lr)
    schedule = torch.optim.lr_scheduler.ExponentialLR(opt, hparams.lr_gamma)
    batch_iter = dataset.iter(hparams.batch_size)

    for step in trange(hparams.steps):
        model_loss = get_loss(model, batch_iter)
        snapshot = model.parameter_snapshot()
        reg_loss = snapshot.heads.mean() * hparams.reg_coeff

        opt.zero_grad()
        loss = model_loss + reg_loss
        loss.backward()
        opt.step()
        schedule.step()

        log_wandb(
            step,
            model.parameter_snapshot(),
            loss=loss.item(),
            model_loss=model_loss.item(),
            reg_loss=reg_loss.item(),
            lr=schedule.get_last_lr()[0],
        )

    if hypers.wandb_enable:
        run.finish()
    return model


# %% [markdown]


base_hypers = TrainHparams(
    transformer_hparams=MixingTransformerHparams(mixer_hparams=MixerHparams(init_p=0.5)),
    steps=80,
    lr=0.1,
    lr_gamma=0.995,
)

final_params = {}

fig, axs = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)

for i, is_direct in enumerate([True, False]):
    for j, reg_coeff in enumerate([0.1, 1, 10]):
        hypers = dataclasses.replace(base_hypers, direct_out=is_direct, reg_coeff=reg_coeff)
        trained_model = train_loop(hypers)

        snap = trained_model.parameter_snapshot()
        final_params[(is_direct, reg_coeff)] = snap


# %%

fig, axs = plt.subplots(2, 3, figsize=(8, 6), sharex=True, sharey=True)
for i, is_direct in enumerate([True, False]):
    for j, reg_coeff in enumerate([0.1, 1, 10]):
        snap = final_params[(is_direct, reg_coeff)]
        head_final_vals = snap.heads[0, :, :].detach().cpu().numpy()
        axs[i, j].matshow(head_final_vals, cmap="Reds", vmin=0, vmax=1)

        if i == 1:
            axs[i, j].set_xlabel("head")
            # axs[i, j].set_xticks(range(12))
            axs[i, j].xaxis.set_ticks_position("bottom")
        if j == 0:
            axs[i, j].set_ylabel("layer")
            # axs[i, j].set_yticks(range(12))


fig.suptitle("Importance of heads for direct paths (top) and all paths (bottom)")

plt.tight_layout()
plt.savefig("mixing_heads.png")

# %%
