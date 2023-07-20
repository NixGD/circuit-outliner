import dataclasses
import json
import os
from typing import Callable, List, Literal, Optional, TypeVar, overload

import torch
from tqdm import trange

import wandb
from mixing_transformer import (
    DirectOutMixingTransformer,
    IndirectMixingTransformer,
    MixingTransformerHparams,
    ParameterSnapshot,
)
from utils import get_base_model, get_owt_dataset, next_toks

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def lm_loss(logits, labels) -> torch.Tensor:
    """
    The standard Language Modelling loss

    predictions: [batch, seqlen-1, vocab_size]
    labels: [batch, seqlen-1]

    They are offset, so labels[i, j] is the next token for predictions[i, j].
    """
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="mean")


def clasificaiton_loss(logits, labels, subset_toks: torch.Tensor) -> torch.Tensor:
    """
    logits: [batch, seqlen-1, vocab_size]
    labels: [batch, seqlen-1]
    subset_toks is a tensor of token_ids that defines one class of tokens

    This evaluates the logits as a classifier for if the next token is in the subset or not
    """
    prob_within = logits.softmax(-1)[:, :, subset_toks].sum(-1)
    is_within = torch.isin(labels, subset_toks).float()
    return torch.nn.functional.binary_cross_entropy(prob_within, is_within)


@dataclasses.dataclass(frozen=True)
class TrainHparams(MixingTransformerHparams):
    base_model: str = "gpt2-small"
    batch_size: int = 4
    steps: int = 200
    lr: float = 0.1
    lr_gamma: float = 0.995
    direct_out: bool = False
    reg_coeff: float = 1
    wandb_enable: bool = True
    classification_subset: Optional[List[int]] = None


test_hypers = TrainHparams(steps=10, wandb_enable=False)


@dataclasses.dataclass(frozen=True)
class ExperimentInfo:
    """
    JSON serializable information about an experiment, used for caching
    """

    hparams: TrainHparams
    snapshot: ParameterSnapshot
    val_loss: float

    def to_json(self):
        snap = self.snapshot
        to_list = lambda x: x.tolist() if x is not None else x
        return {
            "hparams": dataclasses.asdict(self.hparams),
            "snapshot": {
                "heads": to_list(snap.heads),
                "mlps": to_list(snap.mlps),
                "neurons": to_list(snap.neurons),
            },
            "val_loss": self.val_loss,
        }

    @classmethod
    def from_json(cls, data):
        to_tensor = lambda x: torch.tensor(x) if x is not None else x
        return cls(
            hparams=TrainHparams(**data["hparams"]),
            snapshot=ParameterSnapshot(
                heads=to_tensor(data["snapshot"]["heads"]),
                mlps=to_tensor(data["snapshot"]["mlps"]),
                neurons=to_tensor(data["snapshot"]["neurons"]),
            ),
            val_loss=data["val_loss"],
        )


class Experiment:
    """
    This class wraps a mixing model, handling training + evaluation
    """

    def __init__(self, hparams, train=True) -> None:
        self.hparams = hparams
        self.base_model = get_base_model(hparams.base_model)
        self.model = (DirectOutMixingTransformer if hparams.direct_out else IndirectMixingTransformer)(
            self.base_model, hparams
        )
        self.dataset = get_owt_dataset()
        if train:
            self.train()

    def get_dataset_iter(self, batch_size, steps):
        total_data_points = batch_size * steps * 2
        batch_iter = self.dataset.take(total_data_points).iter(batch_size)
        self.dataset = self.dataset.skip(total_data_points)
        return batch_iter

    def get_loss(self, batch_iter) -> torch.Tensor:
        ref_toks, alt_toks = next_toks(batch_iter), next_toks(batch_iter)
        out = self.model(ref_toks, alt_toks)

        logits, labels = out[:, :-1], ref_toks[:, 1:]

        if self.hparams.classification_subset is not None:
            return clasificaiton_loss(
                logits, labels, torch.tensor(self.hparams.classification_subset, device=logits.device)
            )
        else:
            return lm_loss(logits, labels)

    @staticmethod
    def log_wandb(snapshot, **kwargs):
        to_log = kwargs
        to_log["heads/p/mean"] = snapshot.heads[0].mean().item()
        to_log["heads/q/mean"] = snapshot.heads[1].mean().item() if snapshot.heads.shape[0] > 1 else None
        wandb.log(to_log)

    def train(self):
        H = self.hparams
        run = wandb.init(project="outliner", config=dataclasses.asdict(H), mode=None if H.wandb_enable else "disabled")
        self.model.train()
        opt = torch.optim.Adam(self.model.mixer_parameters(), lr=H.lr)
        schedule = torch.optim.lr_scheduler.ExponentialLR(opt, H.lr_gamma)
        batch_iter = self.get_dataset_iter(H.batch_size, H.steps)
        for step in trange(H.steps):
            model_loss = self.get_loss(batch_iter)
            snap = self.model.parameter_snapshot()

            # the len(w) factor means that the total regularization penalty is twice as large if there's an adversary
            # this keeps protaganist behavior comparable with and without an adversary
            get_reg_loss = lambda w: w.mean() * H.reg_coeff * len(w) if w is not None else torch.tensor(0)
            head_reg_loss, mlp_reg_loss, neuron_reg_loss = [
                get_reg_loss(w) for w in (snap.heads, snap.mlps, snap.neurons)
            ]
            reg_loss = head_reg_loss + mlp_reg_loss + neuron_reg_loss
            loss = model_loss + reg_loss

            self.log_wandb(
                snap,
                loss=loss.item(),
                model_loss=model_loss.item(),
                reg_loss=reg_loss.item(),
                head_reg_loss=head_reg_loss.item(),
                mlp_reg_loss=mlp_reg_loss.item(),
                neuron_reg_loss=neuron_reg_loss.item(),
                lr=schedule.get_last_lr()[0],
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            schedule.step()

        run.finish(quiet=True)  # type: ignore

        return self.model

    def snapshot(self):
        return self.model.parameter_snapshot()

    def get_info(self) -> ExperimentInfo:
        """Returns a json-serializable dictionary"""
        return ExperimentInfo(
            hparams=self.hparams,
            snapshot=self.snapshot(),
            val_loss=self.val_loss(),
        )

    def val_loss(self, steps=10, batch_size=None, tqdm_enabled=True):
        self.model.eval()
        batch_size = self.hparams.batch_size if batch_size is None else batch_size
        batch_iter = self.get_dataset_iter(batch_size, steps)
        with torch.no_grad():
            losses = torch.tensor([self.get_loss(batch_iter).item() for _ in trange(steps, disable=not tqdm_enabled)])
        return losses.mean().item()


def get_gpt_loss(steps=10, batch_size=4, classification_subset=None):
    """The loss of the base model on the same dataset and same loss function."""
    gpt = get_base_model()
    batch_iter = get_owt_dataset().iter(batch_size)
    losses = []
    with torch.no_grad():
        for _ in trange(steps):
            toks = next_toks(batch_iter)
            out = gpt(toks)
            logits, labels = out[:, :-1], toks[:, 1:]

            if classification_subset is not None:
                loss = clasificaiton_loss(logits, labels, torch.tensor(classification_subset, device=logits.device))
            else:
                loss = lm_loss(logits, labels)
            losses.append(loss.item())

    return torch.tensor(losses).mean().item()


def get_maybe_cached(name: str, fn: Callable):
    path = "../data/" + name + ".json"
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        result = fn()
        with open(path, "x") as f:
            json.dump(result, f)
        return result


def run_sweep(name: str, hparam_list: List[TrainHparams]) -> List[ExperimentInfo]:
    path = "../data/" + name + ".json"
    if os.path.isfile(path):
        print(f"Getting cached experiment infos from {path}")
        with open(path, "r") as f:
            data = json.load(f)
            return [ExperimentInfo.from_json(d) for d in data]
    else:
        print(f"No cache found. Running new sweep with {len(hparam_list)} experiments.")
        infos = [Experiment(h, train=True).get_info() for h in hparam_list]
        with open(path, "x") as f:
            json.dump([info.to_json() for info in infos], f)
        return infos
