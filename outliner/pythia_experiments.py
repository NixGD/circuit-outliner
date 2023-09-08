
# %%
import dataclasses
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch

from train import Experiment, ExperimentInfo, TrainHparams, run_sweep
from utils import IMG_FOLDER

from plotting import label_axes

# %%
hparams = TrainHparams(
    base_model="EleutherAI/pythia-160M-deduped",
    dataset_name="conceptofmind/pile_cc",
    init_p=0.8,
    adversarial=True,
    mix_mlps=True,
    wandb_enable=False,
    steps=100,
    reg_coeff=1,
    lr=0.1,
)

# %%
datasets = ["conceptofmind/pile_dm_mathematics", "conceptofmind/pile_enron_emails", "conceptofmind/pile_cc", "conceptofmind/pile_wikipedia_en", "conceptofmind/pile_open_web_text_2"]
hp_list = [hparams.evolve(direct_out=direct, dataset_name=dataset) for direct in [True] for dataset in datasets]
sweep_infos = run_sweep("pythia_ds_and_direct_sweep", hp_list)

# %%

def get_matching_info(experiments: List[ExperimentInfo], **kwargs):
    matches = [
        exp for exp in experiments if all(getattr(exp.hparams, k) == v for k, v in kwargs.items())
    ]
    assert len(matches) == 1, f"Found {len(matches)} matches for {kwargs}"
    return matches[0]


def facet_2d(row_key: str, row_values: List, col_key: str, col_values: List):
    fig, axs = plt.subplots(len(row_values), len(col_values), figsize=(15, 6), sharex=True, sharey=True, squeeze=False)
    for row_idx, row_val in enumerate(row_values):
        for col_idx, col_val in enumerate(col_values):
            ax = axs[row_idx, col_idx]  # type: ignore
            info = get_matching_info(sweep_infos, **{row_key: row_val, col_key: col_val})
            ax.matshow(info.snapshot.heads_and_mlps(diff=True), cmap="RdBu", vmin=-1, vmax=1)

            if row_idx == 0:
                ax.set_title(col_val[19:], pad=10)
            
    label_axes(axs)

    return fig
    

fig = facet_2d("direct_out", [True,], "dataset_name", datasets)
fig.set_size_inches(12,3.3)

fig.suptitle("Pythia importance by dataset", size="x-large")
plt.tight_layout()
plt.savefig(IMG_FOLDER + "pythia-dataset.png")

# %%

# # %%
# pythia_info = pythia_exp.get_info()

# (no_adv_info,) = run_sweep("no_adv", compute_no_adv)

#  = comparison_info.snapshot.heads_and_mlps()
# no_adv_w = no_adv_info.snapshot.heads_and_mlps()

# fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))
# axs[0, 0].matshow(no_adv_w[0], cmap="RdBu", vmin=-1, vmax=1)  # type: ignore
# axs[0, 1].matshow(torch.ones_like(no_adv_w[0]), cmap="Greys", vmin=0, vmax=2)  # type: ignore
# axs[1, 0].matshow(adv_w[0], cmap="RdBu", vmin=-1, vmax=1)  # type: ignore
# axs[1, 1].matshow(adv_w[1], cmap="RdBu_r", vmin=-1, vmax=1)  # type: ignore

# label_axes(axs)

# plt.tight_layout()
# plt.savefig(IMG_FOLDER + "adv vs no adv comparison.png")


# # %%


# pythia_hparams = dataclasses.replace(base_hypers, base_model="pythia-70M")
# pythia_exp = Experiment(pythia_hparams)

# %%
