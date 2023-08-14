# %%
import dataclasses
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange
from transformers import PreTrainedTokenizerBase

from mixing_transformer import (
    IndirectMixingTransformer,
    MixingTransformerHparams,
    ParameterSnapshot,
)
from train import (
    Experiment,
    ExperimentInfo,
    TrainHparams,
    get_gpt_loss,
    get_maybe_cached,
    run_sweep,
)
from utils import IMG_FOLDER, get_base_model, get_dataset, kl_div, next_toks

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
    dataset = get_dataset()
    batch_iter = dataset.iter(batch_size=2)
    ref_toks = next_toks(batch_iter)
    alt_toks = next_toks(batch_iter)

    gpt2 = get_base_model()
    with torch.no_grad():
        gpt_ref_out = gpt2(ref_toks)
        gpt_alt_out = gpt2(alt_toks)

        mixer_p1 = IndirectMixingTransformer(
            gpt2, MixingTransformerHparams(init_p=1, mix_heads=True, mix_mlps=True)
        )
        mix_p1_out = mixer_p1(ref_toks, alt_toks)
        print(
            f"kl div between mix_p1 and ref_out: {kl_div(mix_p1_out, gpt_ref_out).item():.5f}",
        )
        assert torch.allclose(gpt_ref_out, mix_p1_out, atol=1e-3)

        mixer_p0 = IndirectMixingTransformer(
            gpt2, MixingTransformerHparams(init_p=0, mix_heads=True, mix_mlps=True)
        )
        mix_p0_out = mixer_p0(ref_toks, alt_toks)
        print(f"kl div between mix_p1 and ref_out: {kl_div(mix_p0_out, gpt_alt_out).item():.5f}")
        assert kl_div(mix_p0_out, gpt_alt_out) < 0.01


test_basic_mixing()
# %% [markdown]

"""
We can also see how loss increases as we decrease p (and thus decrease the amount of information from the reference input that passes through).
"""


def loss_for_p(p: float, is_direct: bool, steps=4, **kwargs) -> float:
    hparams = TrainHparams(
        init_p=p, mix_mlps=True, direct_out=is_direct, batch_size=4, wandb_enable=False, **kwargs
    )
    exp = Experiment(hparams, train=False)
    return exp.val_loss(steps=steps, tqdm_enabled=False)


def compute_losses_for_p(**kwargs):
    ps = torch.linspace(0, 1, 20).tolist()
    print("Computing indirect losses:")
    indirect_losses = [loss_for_p(p, is_direct=False, **kwargs) for p in tqdm(ps)]
    print("Computing direct losses:")
    direct_losses = [loss_for_p(p, is_direct=True, **kwargs) for p in tqdm(ps)]
    gpt_avg_loss = get_gpt_loss()
    return {
        "ps": ps,
        "indirect_losses": indirect_losses,
        "direct_losses": direct_losses,
        "gpt_avg_loss": gpt_avg_loss,
    }


losses = get_maybe_cached("losses_for_p", compute_losses_for_p)
plt.plot(losses["ps"], losses["indirect_losses"], "-o", label="indirect")
plt.plot(losses["ps"], losses["direct_losses"], "-o", label="indirect")
plt.axhline(losses["gpt_avg_loss"], color="k", label="gpt2")

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

is_direct_options = [True, False]
reg_coeff_options = [0.3, 0.55, 1, 1.8, 3]

base_hypers = TrainHparams(
    init_p=0.8,
    adversarial=True,
    mix_mlps=True,
    wandb_enable=False,
    steps=100,
)
hyper_list = [
    dataclasses.replace(base_hypers, direct_out=is_direct, reg_coeff=reg_coeff)
    for is_direct in is_direct_options
    for reg_coeff in reg_coeff_options
]
# this function runs a sweep of the hparam list, if not cached.
sweep_infos = run_sweep("importance_sweep", hyper_list)

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


def get_matching_info(experiments: List[ExperimentInfo], **kwargs):
    matches = [
        exp for exp in experiments if all(getattr(exp.hparams, k) == v for k, v in kwargs.items())
    ]
    assert len(matches) == 1, f"Found {len(matches)} matches for {kwargs}"
    return matches[0]


fig, axs = plt.subplots(2, 3, figsize=(8, 6), sharex=True, sharey=True)
for i, is_direct in enumerate([True, False]):
    for j, reg_coeff in enumerate([0.3, 1, 3]):
        ax = axs[i, j]  # type: ignore
        info = get_matching_info(sweep_infos, direct_out=is_direct, reg_coeff=reg_coeff)
        ax.matshow(info.snapshot.heads_and_mlps(diff=True), cmap="RdBu", vmin=-1, vmax=1)

        if i == 0:
            ax.set_title(f"reg_coeff={reg_coeff}", pad=10)

label_axes(axs)
fig.suptitle("Importance of heads for direct paths (top) and all paths (bottom)", size="x-large")
plt.tight_layout()
plt.savefig(IMG_FOLDER + "mixing_heads.png")

# %% [markdown]

"""
What happens if we remove the adversary?

As we see below, the set of heads that the protaganist finds is the same without an adversary. It does choose somewhat smaller weights for some heads without an adversary present. 
"""

# with an adversary
comparison_info = get_matching_info(sweep_infos, direct_out=True, reg_coeff=3)


def compute_no_adv():
    hparams = dataclasses.replace(comparison_info.hparams, adversarial=False)
    exp = Experiment(hparams)
    exp.train()
    return [exp.get_info()]


(no_adv_info,) = run_sweep("no_adv", compute_no_adv)

adv_w = comparison_info.snapshot.heads_and_mlps()
no_adv_w = no_adv_info.snapshot.heads_and_mlps()

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))
axs[0, 0].matshow(no_adv_w[0], cmap="RdBu", vmin=-1, vmax=1)  # type: ignore
axs[0, 1].matshow(torch.ones_like(no_adv_w[0]), cmap="Greys", vmin=0, vmax=2)  # type: ignore
axs[1, 0].matshow(adv_w[0], cmap="RdBu", vmin=-1, vmax=1)  # type: ignore
axs[1, 1].matshow(adv_w[1], cmap="RdBu_r", vmin=-1, vmax=1)  # type: ignore

label_axes(axs)

plt.tight_layout()
plt.savefig(IMG_FOLDER + "adv vs no adv comparison.png")

# %% [markdown]
"""
How much information is needed to preserve good performance?

We can get a sense of this by sweeping over the regularization coefficient. For larger regularization coefficients, the network will converge on a solution that preserves less information. We can plot this fronteir by measuring the validation loss compared to the mean p value (averaging across attention heads & mlp layers).

We can compare this frontier to the "flat" frontier as seen above, where we initialize all weights in the network with a particular value of p but don't train them.
"""

# sweeping across regularization coefficients

reg_coeffs = [0.5, 1, 2, 4, 8, 12, 13, 13.5, 14, 16, 20, 28]
frontier_hyper_list = [dataclasses.replace(base_hypers, reg_coeff=c) for c in reg_coeffs]
frontier_indirect_infos = run_sweep("frontier_indirect", frontier_hyper_list)

reg_coeffs = [0.5, 1, 2, 4, 8, 12, 14, 16, 20, 28]
frontier_hyper_list = [
    dataclasses.replace(base_hypers, reg_coeff=c, direct_out=True) for c in reg_coeffs
]
frontier_direct_infos = run_sweep("frontier_direct", frontier_hyper_list)
# %%


def get_mean_p(snapshot: ParameterSnapshot) -> float:
    return (snapshot.heads[0].mean().item() + snapshot.mlps[0].mean().item()) / 2


def frontier_plot(
    flat_ps,
    tens,
    flat_gpt_loss,
    frontier_infos: List[ExperimentInfo],
    ax: Optional[plt.Axes] = None,
    xlabel="Mean p",
    ylabel="Loss",
    title=None,
    legend=False,
    normalize=False,
):
    ax = plt.gca() if ax is None else ax
    if normalize:
        norm = lambda x: (x - flat_gpt_loss) / (tens[0] - flat_gpt_loss)
    else:
        norm = lambda x: x

    ax.plot(flat_ps, norm(torch.tensor(tens)), "-o", label="flat")
    ax.plot(
        [get_mean_p(info.snapshot) for info in frontier_infos],
        norm(torch.tensor([info.val_loss for info in frontier_infos])),
        "-o",
        label="frontier",
    )
    ax.axhline(norm(flat_gpt_loss), color="k", label="gpt2")
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend()


fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

flat_losses = get_maybe_cached("losses_for_p", compute_losses_for_p)
frontier_plot(
    flat_losses["ps"],
    flat_losses["indirect_losses"],
    flat_losses["gpt_avg_loss"],
    frontier_indirect_infos,
    ax=axs[0],
    title="Indirect",
)

frontier_plot(
    flat_losses["ps"],
    flat_losses["direct_losses"],
    flat_losses["gpt_avg_loss"],
    frontier_direct_infos,
    ax=axs[1],
    ylabel=None,
    title="Direct",
    legend=True,
)

plt.tight_layout()
fig.savefig(IMG_FOLDER + "frontiers.png")

# %% [markdown]
"""
What does this show?

The more distributed the behavior, the more 

"""

# %% [markdown]

"""
# Testing on a specific classification subtask
"""

tokenizer: PreTrainedTokenizerBase = get_base_model().tokenizer  # type: ignore
all_tokens = tokenizer.batch_decode(range(tokenizer.vocab_size))
eos_toks_ids = [i for i, s in enumerate(all_tokens) if s.startswith((".", "!", "?", ";"))]

eos_base_hypers = dataclasses.replace(
    base_hypers,
    reg_coeff=0.1,  # the loss diff between p=0 and p=1 is very small, which requires a smaller reg_coeff
    classification_subset=eos_toks_ids,
)
eos_hyper_list = [eos_base_hypers, dataclasses.replace(eos_base_hypers, direct_out=True)]
eos_indirect_info, eos_direct_info = run_sweep("eos_experiment", eos_hyper_list)

eos_gpt_loss = get_gpt_loss(classification_subset=eos_toks_ids)
print(f"GPT's loss: {eos_gpt_loss:.3f}")
print(f"Direct loss: {eos_direct_info.val_loss:.3f}")
print(f"Indirect loss: {eos_indirect_info.val_loss:.3f}")

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 4), squeeze=False)
axs[0, 0].matshow(
    eos_indirect_info.snapshot.heads_and_mlps(diff=True), cmap="RdBu", vmin=-1, vmax=1
)
axs[0, 1].matshow(eos_direct_info.snapshot.heads_and_mlps(diff=True), cmap="RdBu", vmin=-1, vmax=1)
label_axes(axs)
axs[0, 0].set_title("Indirect")
axs[0, 1].set_title("Direct")

plt.tight_layout()
fig.savefig(IMG_FOLDER + "punctuation.png")

# %%

# %%
reg_coeffs = [0.002, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
eos_frontier_hypers = [dataclasses.replace(eos_base_hypers, reg_coeff=c) for c in reg_coeffs]
eos_frontier_indirect_infos = run_sweep("eos_frontier_indirect", eos_frontier_hypers)
reg_coeffs = [0.002, 0.005, 0.01, 0.02, 0.05, 0.5, 1]
eos_frontier_hypers = [
    dataclasses.replace(eos_base_hypers, reg_coeff=c, direct_out=True) for c in reg_coeffs
]
eos_frontier_direct_infos = run_sweep("eos_frontier_direct", eos_frontier_hypers)

# %%

eos_flat_losses = get_maybe_cached(
    "eos_losses_for_p", lambda: compute_losses_for_p(classification_subset=eos_toks_ids)
)
plt.plot(eos_flat_losses["ps"], eos_flat_losses["indirect_losses"], "-o", label="indirect")
plt.plot(eos_flat_losses["ps"], eos_flat_losses["direct_losses"], "-o", label="direct")
plt.axhline(eos_gpt_loss, color="k", label="gpt2")

plt.ylabel("Loss")
plt.xlabel("p")
plt.gcf().set_size_inches(5, 5)
plt.legend()
# %%

fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

frontier_plot(
    eos_flat_losses["ps"],
    eos_flat_losses["indirect_losses"],
    eos_gpt_loss,
    eos_frontier_indirect_infos,
    ax=axs[0],
    title="Indirect",
)
frontier_plot(
    eos_flat_losses["ps"],
    eos_flat_losses["direct_losses"],
    eos_gpt_loss,
    eos_frontier_direct_infos,
    ax=axs[1],
    ylabel=None,
    title="Direct",
    legend=True,
)

plt.tight_layout()
fig.savefig(IMG_FOLDER + "eos_frontiers.png")

# %%

fig, axs = plt.subplots(2, 2, figsize=(6, 6), sharey="row", sharex=True)
normalize = True
frontier_plot(
    flat_losses["ps"],
    flat_losses["indirect_losses"],
    flat_losses["gpt_avg_loss"],
    frontier_indirect_infos,
    ax=axs[0][0],
    title="Indirect",
    xlabel=None,
    normalize=normalize,
)

frontier_plot(
    flat_losses["ps"],
    flat_losses["direct_losses"],
    flat_losses["gpt_avg_loss"],
    frontier_direct_infos,
    ax=axs[0][1],
    ylabel=None,
    title="Direct",
    legend=True,
    xlabel=None,
    normalize=normalize,
)
frontier_plot(
    eos_flat_losses["ps"],
    eos_flat_losses["indirect_losses"],
    eos_gpt_loss,
    eos_frontier_indirect_infos,
    ax=axs[0][0],
    normalize=normalize,
)
frontier_plot(
    eos_flat_losses["ps"],
    eos_flat_losses["direct_losses"],
    eos_gpt_loss,
    eos_frontier_direct_infos,
    ax=axs[0][1],
    ylabel=None,
    normalize=normalize,
)

plt.tight_layout()
fig.savefig(IMG_FOLDER + "frontier_comparisons.png")
# %%
