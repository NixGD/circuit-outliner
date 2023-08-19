from typing import List, Optional

import matplotlib.pyplot as plt
import torch

from mixing_transformer import ParameterSnapshot
from train import ExperimentInfo


def label_axes(axs):
    for ax in axs[:, 0]:
        ax.set_ylabel("layer")
        ax.set_yticks(range(12))
    for ax in axs[-1, :]:
        ax.set_xlabel("head")
        ax.set_xticks(range(13))
        ax.set_xticklabels(list(range(12)) + ["M"])
        ax.xaxis.set_ticks_position("bottom")


def get_mean_p(snapshot: ParameterSnapshot) -> float:
    return (snapshot.heads[0].mean().item() + snapshot.mlps[0].mean().item()) / 2


def frontier_plot(
    flat_ps,
    flat_losses,
    flat_gpt_loss: Optional[float],
    frontier_infos: List[ExperimentInfo],
    ax: Optional[plt.Axes] = None,
    xlabel="Mean p",
    ylabel="Loss",
    title=None,
    legend=False,
    normalize=False,
    color="black",
):
    ax = plt.gca() if ax is None else ax
    if normalize:
        norm = lambda x: (x - flat_losses[-1]) / (flat_losses[0] - flat_losses[-1])
    else:
        norm = lambda x: x

    ax.plot(flat_ps, norm(torch.tensor(flat_losses)), ls="dashed", label="flat", ms=5, color=color)
    ax.plot(
        [get_mean_p(info.snapshot) for info in frontier_infos],
        norm(torch.tensor([info.val_loss for info in frontier_infos])),
        "-o",
        label="frontier",
        ms=5,
        color=color,
    )
    if flat_gpt_loss is not None:
        ax.axhline(norm(flat_gpt_loss), color="k", label="gpt2")
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend()


def frontier_fig_facet_by_direct(flat_losses, indirect_infos, direct_infos):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    frontier_plot(
        flat_losses["ps"],
        flat_losses["indirect_losses"],
        flat_losses["gpt_avg_loss"],
        indirect_infos,
        ax=axs[0],
        title="Indirect",
    )

    frontier_plot(
        flat_losses["ps"],
        flat_losses["direct_losses"],
        flat_losses["gpt_avg_loss"],
        direct_infos,
        ax=axs[1],
        ylabel=None,
        title="Direct",
        legend=True,
    )

    fig.tight_layout()
    return fig
