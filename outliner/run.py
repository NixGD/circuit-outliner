# %%
import dataclasses
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from mixer import MixerHparams
from mixing_transformer import (DirectOutMixingTransformer,
                                IndirectMixingTransformer,
                                MixingTransformerHparams, ParameterSnapshot)
from train import TrainHparams, get_loss, lm_loss, train_loop
from utils import DEVICE, get_base_model, get_owt_dataset, kl_div

# %% [markdown]

"""
# Introduction

The basic operation that MixingTransformer impliments is to test a path patching hypothesis by mixing together the activations from two different inputs.

For instance, let us say that we are just mixing the output of one attention head. On our reference input it outputs $x_{ref}$, while on the alternate input it outputs $x_{alt}$. When running on the alternate input we replace $x_{ref}$ with $p*x_{ref} + (1-p)*x_{alt}$.

This is, in essence, an ablation experiment that keeps the attention head 'on distribution' in a meaningful sense. It thus gives an indication of the importance of the attention head at influencing the output by distinguishing within-distribution inputs. 

We can either to this for all paths between our attention head and the output (indirect mixing) or only along the direct path to the output (direct output mixing).

In general we do this for many components at once, instead of just a single attention head. Each component (attention head, mlp layer, or individual neuron) has a separate $p$ parameter which we learned with gradient decent.
"""

# %% [markdown]

"""
# Simple MixingTransformer tests

With p=1, our mixing transformer is just running on the ref input.

With p=0, our mixing transformer is just running on the alt input -- with the execption of the direct path through the residual stream from embed to unemebed. Thus while the exact outputs aren't 
"""


def test_basic_mixing():
    dataset = get_owt_dataset()
    batch_iter = dataset.iter(batch_size=2)
    ref_toks = torch.stack(next(batch_iter)["toks"])
    alt_toks = torch.stack(next(batch_iter)["toks"])

    gpt2 = get_base_model()
    with torch.no_grad():
        gpt_ref_out = gpt2(ref_toks)
        gpt_alt_out = gpt2(alt_toks)

        mixer_p1 = IndirectMixingTransformer(
            gpt2, MixingTransformerHparams(mixer_hparams=MixerHparams(init_p=1), mix_heads=True, mix_mlps=True)
        )
        mix_p1_out = mixer_p1(ref_toks, alt_toks)
        print("kl div between mix_p1 and ref_out: ", kl_div(mix_p1_out, gpt_ref_out).item())
        assert torch.allclose(gpt_ref_out, mix_p1_out, atol=1e-3)

        mixer_p0 = IndirectMixingTransformer(
            gpt2, MixingTransformerHparams(mixer_hparams=MixerHparams(init_p=0), mix_heads=True, mix_mlps=True)
        )
        mix_p0_out = mixer_p0(ref_toks, alt_toks)
        print("kl div between mix_p1 and ref_out: ", kl_div(mix_p0_out, gpt_alt_out).item())
        assert kl_div(mix_p0_out, gpt_alt_out) < 0.01


test_basic_mixing()
# %% [markdown]

"""
We can also see how loss increases as we decrease p (and thus decrease the amount of information from the reference input that passes through).
"""


def get_losses_for_p(p: float | torch.Tensor, steps=10, loss_fn=lm_loss) -> Tuple[float, float]:
    gpt2 = get_base_model()
    hparams = MixingTransformerHparams(mixer_hparams=MixerHparams(init_p=p), mix_mlps=True)
    mixer_ind = IndirectMixingTransformer(gpt2, hparams)
    mixer_dir = DirectOutMixingTransformer(gpt2, hparams)

    dataset = get_owt_dataset()
    batch_iter = dataset.iter(batch_size=4)
    losses_ind, losses_dir = [], []
    with torch.no_grad():
        for _ in trange(steps):
            losses_ind.append(get_loss(mixer_ind, batch_iter, loss_fn=lm_loss).item())
            losses_dir.append(get_loss(mixer_dir, batch_iter, loss_fn=lm_loss).item())

    mean = lambda l: sum(l) / len(l)
    return mean(losses_ind), mean(losses_dir)


def get_gpt_avg_loss():
    gpt2 = get_base_model()
    dataset = get_owt_dataset()
    batch_iter = dataset.iter(batch_size=4)
    with torch.no_grad():
        losses = [gpt2(torch.stack(next(batch_iter)["toks"]), return_type="loss").item() for _ in trange(10)]
    return sum(losses) / len(losses)


ps = torch.arange(0, 1.01, 0.2)
losses = torch.tensor([get_losses_for_p(p) for p in ps])

plt.plot(ps, losses[:, 0], "-o", label="indirect")
plt.plot(ps, losses[:, 1], "-o", label="direct")
plt.axhline(get_gpt_avg_loss(), color="k", label="gpt2")

plt.ylabel("Loss")
plt.xlabel("p")
plt.gcf().set_size_inches(5, 5)
plt.legend()

# %% [markdown]

"""
# Importance over entire dataset

Now lets see which parts of the model are most important! We'll do this for next token prediction on the entire dataset. We would expect, therefore, that ~all of the model is helpful when considering indirect connections, and that later parts of the model will be more helpful when considering direct-to-output connections.

We'll run a sweep over {direct, indirect} and {0.1, 1, 10} regularization coefficient. Higher regularization coefficients make it more expensive to include components. With a regularization coefficient of 1, including all heads and all mlps adds 2 to the loss (comparible to gpt small's natural loss of ~3).

We'll also add an adversary to the above experiment. This is a player that can increase the value of p for certain components -- effectively force them to not be mixed. However, the adversary is trying to _maximize_ the model loss. This is helpful to ensure that if there are multiple heads which cancel eachother out we include all of them.

Components in red are included by the adversary.

"""

base_hypers = TrainHparams(
    transformer_hparams=MixingTransformerHparams(
        mixer_hparams=MixerHparams(init_p=0.5, adversarial=True), mix_mlps=True
    ),
    steps=75,
)

final_params: Dict[Tuple, ParameterSnapshot] = {}

for i, is_direct in enumerate([True, False]):
    for j, reg_coeff in enumerate([0.3, 1, 3]):
        hypers = dataclasses.replace(base_hypers, direct_out=is_direct, reg_coeff=reg_coeff)
        trained_model = train_loop(hypers)

        snap = trained_model.parameter_snapshot()
        final_params[(is_direct, reg_coeff)] = snap

# %%


def label_axes(axs):
    for ax in axs[:, 0]:
        ax.set_ylabel("layer")
        ax.set_yticks(range(12))
    for ax in axs[-1, :]:
        ax.set_xlabel("head")
        ax.set_xticks(range(13))
        ax.set_xticklabels(list(range(12)) + ["M"])
        ax.xaxis.set_ticks_position("bottom")


fig, axs = plt.subplots(2, 3, figsize=(8, 6), sharex=True, sharey=True)
for i, is_direct in enumerate([True, False]):
    for j, reg_coeff in enumerate([0.3, 1, 3]):
        ax = axs[i, j]  # type: ignore
        w = final_params[(is_direct, reg_coeff)].heads_and_mlps()
        ax.matshow(w[0] - w[1], cmap="RdBu", vmin=-1, vmax=1)

        if i == 0:
            ax.set_title(f"reg_coeff={reg_coeff}", pad=10)

label_axes(axs)
fig.suptitle("Importance of heads for direct paths (top) and all paths (bottom)", size="x-large")
plt.tight_layout()
plt.savefig("mixing_heads.png")

# %% [markdown]

"""
What happens if we remove the adversary?

As we see below, the set of heads that the protaganist finds is the same without an adversary. It does choose somewhat smaller weights for some heads without an adversary present. 
"""

no_adv_hypers = TrainHparams(
    transformer_hparams=MixingTransformerHparams(
        mixer_hparams=MixerHparams(init_p=0.5, adversarial=False), mix_mlps=True
    ),
    reg_coeff=3,
    direct_out=True,
)
no_adv_model = train_loop(no_adv_hypers)

# %%

adv_w = final_params[(True, 3)].heads_and_mlps()
no_adv_w = final_params[(True, 3)].heads_and_mlps()

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))
axs[0, 0].matshow(no_adv_w[0], cmap="RdBu", vmin=-1, vmax=1)
axs[0, 1].matshow(torch.ones_like(no_adv_w[0]), cmap="Greys", vmin=0, vmax=2)
axs[1, 0].matshow(adv_w[0], cmap="RdBu", vmin=-1, vmax=1)
axs[1, 1].matshow(adv_w[1], cmap="RdBu_r", vmin=-1, vmax=1)

label_axes(axs)

plt.tight_layout()
plt.savefig("adv vs no adv comparison.png")

# %% [markdown]

"""
# Testing on a specific classification subtask
"""

tokenizer = get_base_model().tokenizer
all_tokens = tokenizer.batch_decode(range(tokenizer.vocab_size))
eos_toks_ids = [i for i, s in enumerate(all_tokens) if s.startswith((".", "!", "?", ";"))]
eos_toks_ids = torch.tensor(eos_toks_ids, device=DEVICE)


def eos_loss(logits, labels) -> torch.Tensor:
    """
    logits: [batch, seqlen-1, vocab_size]
    labels: [batch, seqlen-1]
    """
    prob_of_eos = logits.softmax(-1)[:, :, eos_toks_ids].sum(-1)
    is_eos = torch.isin(labels, eos_toks_ids).float()
    return torch.nn.functional.binary_cross_entropy(prob_of_eos, is_eos)


def get_gpt_eos_loss():
    gpt2 = get_base_model()
    dataset = get_owt_dataset()
    batch_iter = dataset.iter(batch_size=40)
    losses = []
    with torch.no_grad():
        for _ in trange(2):
            toks = torch.stack(next(batch_iter)["toks"])
            out = gpt2(toks)
            losses.append(eos_loss(out[:, :-1], toks[:, 1:]).item())

    return sum(losses) / len(losses)


gpt_eos_loss = get_gpt_eos_loss()
print(gpt_eos_loss)
# %%

eos_hypers = TrainHparams(
    transformer_hparams=MixingTransformerHparams(
        mixer_hparams=MixerHparams(init_p=0.5, adversarial=True), mix_mlps=True
    ),
    reg_coeff=0.1,
    direct_out=True,
    loss_fn=eos_loss,
)
eos_model = train_loop(eos_hypers)

snap = eos_model.parameter_snapshot()
w = snap.heads_and_mlps()
plt.matshow(w[0] - w[1], cmap="RdBu", vmin=-1, vmax=1)

# %%

eos_model_ind = train_loop(dataclasses.replace(eos_hypers, direct_out=False))

# %%

snap = eos_model_ind.parameter_snapshot()
w = snap.heads_and_mlps()
plt.matshow(w[0] - w[1], cmap="RdBu", vmin=-1, vmax=1)

# %%


def eval_model(model, loss_fn=lm_loss):
    model.eval()
    batch_iter = get_owt_dataset().skip(1000).iter(32)
    with torch.no_grad():
        losses = torch.stack([get_loss(model, batch_iter, loss_fn) for _ in trange(5)])
        return losses.mean().item()


base_hypers = TrainHparams(
    transformer_hparams=MixingTransformerHparams(
        mixer_hparams=MixerHparams(init_p=0.8, adversarial=False), mix_mlps=True
    ),
    direct_out=False,
    steps=75,
)

flat_ps = torch.arange(0, 1.01, 0.2)
losses_for_flat_ps = torch.tensor([get_losses_for_p(p) for p in flat_ps])
gpt_avg_loss = get_gpt_avg_loss()

# %%

reg_coeffs = [0.25, 0.5, 1, 2, 3, 4, 5, 7, 10]
losses = []
avg_ps = []

for reg_coeff in reg_coeffs:
    hypers = dataclasses.replace(base_hypers, reg_coeff=reg_coeff)
    model = train_loop(hypers)
    loss = eval_model(model)
    losses.append(loss)

    snap = model.parameter_snapshot()
    # p + q. Technically p + q - pq would be better, but in practice they are equal
    avg_p = (snap.heads.sum(0).mean() + snap.mlps.sum(0).mean()).item() / 2
    avg_ps.append(avg_p)
    print(f"{reg_coeff=} {loss=} {avg_p=}")

    fig, ax = plt.subplots(1, 1)
    ax.plot(ps, losses_for_flat_ps[:, 0], "-o", label="indirect")
    ax.axhline(gpt_avg_loss, color="k", label="gpt2")
    ax.plot(avg_ps, losses)
    fig.savefig("indirect_frontier.png")

# %%
with open("../data/frontier_indirect.json", "w") as f:
    json.dump({"reg_coeffs": reg_coeffs, "losses": losses, "avg_ps": avg_ps}, f)
# %%
