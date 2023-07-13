import itertools
from dataclasses import dataclass

import einops
import torch
from torch import nn
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from mixer import Mixer, MixerHparams


@dataclass(frozen=True)
class MixingTransformerHparams:
    mix_heads: bool = True
    mix_neurons: bool = False  # 1 parameter per neuron
    mix_mlps: bool = False  # 1 parameter per layer
    mixer_hparams: MixerHparams = MixerHparams()


class ParameterLog:
    """
    Only supports head parameters for now, until I decide if I want to commit to this storage approach or not.

    head_params: [step, player, layer, head]
    """

    def init():
        self.head_params = torch.Tensor()

    def append(params):
        """
        params: [player, layer, head]
        """
        self.head_params = torch.cat([self.head_params, params], dim=0)


class MixingTransformer(nn.Module):
    def __init__(
        self,
        base_model: HookedTransformer,
        hparams: MixingTransformerHparams = MixingTransformerHparams(),
    ):
        super().__init__()
        self.base_model = base_model
        self.cfg = self.base_model.cfg
        self.hparams = hparams

        self.mixers = {}  # will be in model order
        for l in range(self.cfg.n_layers):
            if self.hparams.mix_heads:
                self.mixers[f"blocks.{l}.attn.hook_z"] = Mixer(
                    (self.cfg.n_heads, 1), type="head", hparams=self.hparams.mixer_hparams
                )
            if self.hparams.mix_neurons:
                assert not self.hparams.mix_mlp
                self.mixers[f"blocks.{l}.mlp.hook_post"] = Mixer(
                    (self.cfg.d_mlp), type="neuron", hparams=self.hparams.mixer_hparams
                )
            if self.hparams.mix_mlps:
                assert not self.hparams.mix_neurons
                self.mixers[f"blocks.{l}.hook_mlp_out"] = Mixer((1,), type="mlp", hparams=self.hparams.mixer_hparams)

    def forward(self, ref_in, alt_in, **kwargs):
        ins = torch.cat([ref_in, alt_in], axis=0)
        ref_idxs = torch.arange(len(ins)) < len(ref_in)

        def hook(x, hook: HookPoint):
            mixer = self.mixers[hook.name]
            x[ref_idxs] = mixer(x[ref_idxs], x[~ref_idxs])
            return x

        is_mixed = lambda name: name in self.mixers
        out = self.base_model.run_with_hooks(ins, fwd_hooks=[(is_mixed, hook)], **kwargs)
        return out[ref_idxs]

    def mixer_parameters(self):
        return itertools.chain.from_iterable(m.parameters() for m in self.mixers.values())

    def log_parameters(self, log: ParameterLog()):
        ps = torch.cat([m.p for m in self.mixers.values()], dim=0)
        if self.hparams.mixer_hparams.adversarial:
            qs = torch.cat([m.q for m in self.mixers.values()], dim=0)
            log.append(torch.stack([ps, qs], dim=0))
        else:
            log.append(ps.unsqueeze(0))


class DirectOutMixingTransformer(nn.Module):
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

        self.mixer = Mixer(
            (self.cfg.n_layers, self.cfg.n_heads, 1, 1), type="all_heads", hparams=self.hparams.mixer_hparams
        )

    def forward(self, ref_in, alt_in, **kwargs):
        ins = torch.cat([ref_in, alt_in], axis=0)
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

    def log_parameters(self, log: ParameterLog()):
        ps = self.mixer.p
        if self.hparams.mixer_hparams.adversarial:
            qs = self.mixer.q
            log.append(torch.stack([ps, qs], dim=0))
        else:
            log.append(ps.unsqueeze(0))
