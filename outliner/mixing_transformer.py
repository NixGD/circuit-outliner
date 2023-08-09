import itertools
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import einops
import torch
from torch import nn
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

from mixer import LocType, Mixer, MixerHparams


@dataclass(frozen=True)
class MixingTransformerHparams(MixerHparams):
    mix_heads: bool = True
    mix_neurons: bool = False  # 1 parameter per neuron
    mix_mlps: bool = False  # 1 parameter per layer


@dataclass
class ParameterSnapshot:
    heads: Optional[torch.Tensor] = None  # [adversary, layer, heads]
    neurons: Optional[torch.Tensor] = None  # [adversary, layer, neuron]
    mlps: Optional[torch.Tensor] = None  # [adversary, layer]

    def heads_and_mlps(self, diff=False):
        assert self.heads is not None and self.mlps is not None
        w = torch.cat((self.heads, self.mlps.unsqueeze(2)), dim=2).detach().cpu()
        return w[0] - w[1] if diff else w


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
                    (self.cfg.n_heads, 1), "head", hparams=self.hparams
                )
            if self.hparams.mix_neurons:
                assert not self.hparams.mix_mlps
                assert self.cfg.d_mlp is not None
                self.mixers[f"blocks.{l}.mlp.hook_post"] = Mixer(
                    (self.cfg.d_mlp,), "neuron", hparams=self.hparams
                )
            if self.hparams.mix_mlps:
                assert not self.hparams.mix_neurons
                self.mixers[f"blocks.{l}.hook_mlp_out"] = Mixer((1,), "mlp", hparams=self.hparams)

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
            ms = [m for m in self.mixers.values() if m.loc_type == type]
            if len(ms) == 0:
                return None
            ps = torch.stack([m.p for m in ms], dim=0).squeeze()
            if self.hparams.adversarial:
                qs = torch.stack([m.q for m in ms], dim=0).squeeze()
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
        self.mixers = {}
        if self.hparams.mix_heads:
            self.mixers["head"] = Mixer(
                (self.cfg.n_layers, self.cfg.n_heads, 1), "head", hparams=self.hparams
            )
        if self.hparams.mix_neurons:
            assert not self.hparams.mix_mlps
            raise NotImplementedError
        if self.hparams.mix_mlps:
            assert not self.hparams.mix_neurons
            self.mixers["mlp"] = Mixer((self.cfg.n_layers, 1), "mlp", hparams=self.hparams)

    def _residual_diff_for_heads(self, cache: ActivationCache, ref_idxs: torch.Tensor):
        cache.compute_head_results()
        head_results = cache.stack_head_results()
        head_results = einops.rearrange(
            head_results,
            "(layer head) batch seqlen emb -> batch seqlen layer head emb",
            layer=self.cfg.n_layers,
            head=self.cfg.n_heads,
        )

        # we mix head outputs from the ref batch with head outputs from the alt batch
        mixed_head_results = self.mixers["head"](head_results[ref_idxs], head_results[~ref_idxs])
        return mixed_head_results.sum((2, 3)) - head_results[ref_idxs].sum((2, 3))

    def _residual_diff_for_mlps(self, cache: ActivationCache, ref_idxs: torch.Tensor):
        # [batch, seqlen, layer, emb]
        mlp_results = torch.stack(
            [cache[f"blocks.{l}.hook_mlp_out"] for l in range(self.cfg.n_layers)], dim=2
        )
        mixed_mlp_results = self.mixers["mlp"](mlp_results[ref_idxs], mlp_results[~ref_idxs])
        return mixed_mlp_results.sum(2) - mlp_results[ref_idxs].sum(2)

    def forward(self, ref_in, alt_in, **kwargs):
        ins = torch.cat([ref_in, alt_in], dim=0)
        ref_idxs = torch.arange(len(ins)) < len(ref_in)

        _, cache = self.base_model.run_with_cache(ins, **kwargs)
        pre_final_ln = cache[f"blocks.{self.cfg.n_layers-1}.hook_resid_post"][
            ref_idxs
        ]  # [ref_batch, seqlen, emb]
        if self.hparams.mix_heads:
            pre_final_ln += self._residual_diff_for_heads(cache, ref_idxs)
        if self.hparams.mix_mlps:
            pre_final_ln += self._residual_diff_for_mlps(cache, ref_idxs)
        out = self.base_model.unembed(self.base_model.ln_final(pre_final_ln))
        return out

    def mixer_parameters(self):
        return itertools.chain.from_iterable(m.parameters() for m in self.mixers.values())

    def parameter_snapshot(
        self,
    ) -> ParameterSnapshot:
        def stack_params(type: str) -> Optional[torch.Tensor]:
            m: Optional[Mixer] = self.mixers.get(type)
            if m is None:
                return None
            if m.adversarial:
                return torch.stack([m.p.squeeze(), m.q.squeeze()], dim=0)
            else:
                return m.p.squeeze().unsqueeze(0)

        return ParameterSnapshot(
            heads=stack_params("head"),
            neurons=stack_params("neuron"),
            mlps=stack_params("mlp"),
        )
