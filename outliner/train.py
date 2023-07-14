import dataclasses

import torch
from tqdm import trange
from transformer_lens import HookedTransformer

import wandb
from mixing_transformer import (DirectOutMixingTransformer,
                                IndirectMixingTransformer, MixingTransformer,
                                MixingTransformerHparams)
from utils import DEVICE, get_owt_dataset, get_base_model


@dataclasses.dataclass(frozen=True)
class TrainHparams:
    base_model: str = "gpt2-small"
    transformer_hparams: MixingTransformerHparams = MixingTransformerHparams()
    batch_size: int = 4
    steps: int = 200
    lr: float = 0.1
    lr_gamma: float = 0.995
    direct_out: bool = False
    reg_coeff: float = 1
    wandb_enable: bool = True


test_hypers = TrainHparams(steps=10, wandb_enable=False)


def get_mixer_model(hparams=TrainHparams) -> MixingTransformer:
    base_model = get_base_model(hparams.base_model)
    if hparams.direct_out:
        return DirectOutMixingTransformer(base_model, hparams=hparams.transformer_hparams)
    else:
        return IndirectMixingTransformer(base_model, hparams=hparams.transformer_hparams)


def lm_loss(logits, labels) -> torch.Tensor:
    """
    computes the normal lm loss

    predictions: [batch, seqlen-1, vocab_size]
    labels: [batch, seqlen-1]

    They are offset, so labels[i, j] is the next token for predictions[i, j].
    """
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="mean")


def get_loss(model, batch_iter, loss_fn=lm_loss) -> torch.Tensor:
    ref_toks = torch.stack(next(batch_iter)["toks"])
    alt_toks = torch.stack(next(batch_iter)["toks"])
    out = model.forward(ref_toks, alt_toks)
    return loss_fn(out[:, :-1], ref_toks[:, 1:])


def log_wandb(step, snapshot, **kwargs):
    to_log = kwargs
    to_log["heads/p/mean"] = snapshot.heads[0].mean().item()
    to_log["heads/q/mean"] = snapshot.heads[1].mean().item() if snapshot.heads.shape[0] > 1 else None
    wandb.log(to_log)


def train_loop(hparams: TrainHparams):
    run = wandb.init(
        project="outliner", config=dataclasses.asdict(hparams), mode=None if hparams.wandb_enable else "offline"
    )
    model = get_mixer_model(hparams)
    opt = torch.optim.Adam(model.mixer_parameters(), lr=hparams.lr)
    schedule = torch.optim.lr_scheduler.ExponentialLR(opt, hparams.lr_gamma)

    dataset = get_owt_dataset()
    batch_iter = dataset.iter(hparams.batch_size)

    for step in trange(hparams.steps):
        model_loss = get_loss(model, batch_iter)
        snap = model.parameter_snapshot()

        # the len(w) factor means that the total regularization penalty is twice as large if there's an adversary
        # this keeps protaganist behavior comparable with and without an adversary
        get_reg_loss = lambda w: w.mean() * hparams.reg_coeff * len(w) if w is not None else torch.tensor(0)
        head_reg_loss, mlp_reg_loss, neuron_reg_loss = [get_reg_loss(w) for w in (snap.heads, snap.mlps, snap.neurons)]
        reg_loss = head_reg_loss + mlp_reg_loss + neuron_reg_loss
        loss = model_loss + reg_loss

        log_wandb(
            step,
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
    return model
