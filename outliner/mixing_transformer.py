import itertools
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import einops
import torch
from torch import nn
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

from mixer import LocType, Mixer, MixerHparams


@dataclass(frozen=True)
class MixingTransformerHparams:
    mix_heads: bool = True
    mix_neurons: bool = False  # 1 parameter per neuron
    mix_mlps: bool = False  # 1 parameter per layer
    mixer_hparams: MixerHparams = MixerHparams()


@dataclass
class ParameterSnapshot:
    heads: Optional[torch.Tensor] = None  # [adversary, layer, heads]
    neurons: Optional[torch.Tensor] = None  # [adversary, layer, neuron]
    mlps: Optional[torch.Tensor] = None  # [adversary, layer]


class MixingTransformer(nn.Module):
    def forward(self):
        raise NotImplementedError

    def mixer_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError

    def parameter_snapshot(self) -> ParameterSnapshot:
        raise NotImplementedError


class IndirectMixingTransformer(MixingTransformer):
    def __init__(
        self,
        base_model: HookedTransformer,
        hparams: MixingTransformerHparams = MixingTransformerHparams(),
    ):
        super().__init__()
        self.base_model = base_model
        self.cfg: HookedTransformerConfig = self.base_model.cfg
        self.hparams = hparams

        self.mixers = {}  # will be in model order
        for l in range(self.cfg.n_layers):
            if self.hparams.mix_heads:
                self.mixers[f"blocks.{l}.attn.hook_z"] = Mixer(
                    (self.cfg.n_heads, 1), "head", hparams=self.hparams.mixer_hparams
                )
            if self.hparams.mix_neurons:
                assert not self.hparams.mix_mlps
                assert self.cfg.d_mlp is not None
                self.mixers[f"blocks.{l}.mlp.hook_post"] = Mixer(
                    (self.cfg.d_mlp,), "neuron", hparams=self.hparams.mixer_hparams
                )
            if self.hparams.mix_mlps:
                assert not self.hparams.mix_neurons
                self.mixers[f"blocks.{l}.hook_mlp_out"] = Mixer((1,), "mlp", hparams=self.hparams.mixer_hparams)

    def forward(self, ref_in: torch.Tensor, alt_in: torch.Tensor, **kwargs) -> torch.Tensor:
        ins = torch.cat([ref_in, alt_in], dim=0)
        ref_idxs = torch.arange(len(ins)) < len(ref_in)

        def hook(x, hook: HookPoint):
            mixer = self.mixers[hook.name]
            x[ref_idxs] = mixer(x[ref_idxs], x[~ref_idxs])
            return x

        is_mixed = lambda name: name in self.mixers
        out = self.base_model.run_with_hooks(ins, fwd_hooks=[(is_mixed, hook)], **kwargs)
        return out[ref_idxs]

    def mixer_parameters(self):
        """Iterator of parameters to optimize, as in `nn.Module.parameters()`"""
        return itertools.chain.from_iterable(m.parameters() for m in self.mixers.values())

    def parameter_snapshot(self) -> ParameterSnapshot:
        """for logging and regularization calculations"""

        def stack_params(type: str) -> Optional[torch.Tensor]:
            ms = [m for m in self.mixers.values() if m.type == type]
            if len(ms) == 0:
                return None
            ps = torch.stack([m.p for m in ms], dim=0)
            if self.hparams.mixer_hparams.adversarial:
                qs = torch.stack([m.q for m in ms], dim=0)
                return torch.stack([ps, qs], dim=0)
            else:
                return ps.unsqueeze(0)

        return ParameterSnapshot(
            heads=stack_params("head"),
            neurons=stack_params("neuron"),
            mlps=stack_params("mlp"),
        )


class DirectOutMixingTransformer(MixingTransformer):
    def __init__(
        self,
        base_model: HookedTransformer,
        hparams: MixingTransformerHparams = MixingTransformerHparams(),
    ):
        super().__init__()
        self.base_model = base_model
        self.cfg = self.base_model.cfg
        self.hparams = hparams
        # TODO: generalize
        if not self.hparams.mix_heads or self.hparams.mix_mlps or self.hparams.mix_neurons:
            raise NotImplementedError

        self.mixer = Mixer((self.cfg.n_layers, self.cfg.n_heads, 1, 1), "head", hparams=self.hparams.mixer_hparams)

    def forward(self, ref_in, alt_in, **kwargs):
        ins = torch.cat([ref_in, alt_in], dim=0)
        ref_idxs = torch.arange(len(ins)) < len(ref_in)

        _, cache = self.base_model.run_with_cache(ins, **kwargs)
        cache.compute_head_results()
        head_results = cache.stack_head_results()
        head_results = einops.rearrange(
            head_results,
            "(layer head) batch seqlen emb -> batch layer head seqlen emb",
            layer=self.cfg.n_layers,
            head=self.cfg.n_heads,
        )

        # we mix head outputs from the ref batch with head outputs from the alt batch
        mixed_head_results = self.mixer(head_results[ref_idxs], head_results[~ref_idxs])
        pre_final_ln = cache[f"blocks.{self.cfg.n_layers-1}.hook_resid_post"][ref_idxs]  # [ref_batch, seqlen, emb]

        # new_residual = ref_residual - ref_residual_from_heads + mixed_residual_from_heads
        new_pre_final_ln = pre_final_ln - head_results[ref_idxs].sum((1, 2)) + mixed_head_results.sum((1, 2))
        out = self.base_model.unembed(self.base_model.ln_final(new_pre_final_ln))
        return out

    def mixer_parameters(self):
        return self.mixer.parameters()

    def parameter_snapshot(
        self,
    ) -> ParameterSnapshot:
        heads = (
            torch.stack([self.mixer.p, self.mixer.q], 0)
            if self.hparams.mixer_hparams.adversarial
            else self.mixer.p.unsqueeze(0)
        )
        return ParameterSnapshot(
            heads=heads,
            neurons=None,
            mlps=None,
        )
