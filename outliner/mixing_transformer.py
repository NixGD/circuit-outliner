from dataclasses import dataclass

import torch
from torch import nn
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from mixer import Mixer, MixerHparams


@dataclass(frozen=True)
class MixingTransformerHparams:
    mix_heads: bool = True
    mix_neurons: bool = False # 1 parameter per neuron
    mix_mlps: bool = False # 1 parameter per layer
    mixer_hparams: MixerHparams = MixerHparams()


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

        self.mixers = {} # will be in model order
        for l in range(self.cfg.n_layers):
            if self.hparams.mix_heads:
                self.mixers[f"blocks.{l}.attn.hook_z"] = Mixer((self.cfg.n_heads, 1), type="head", hparams=self.hparams.mixer_hparams)
            if self.hparams.mix_neurons:
                assert not self.hparams.mix_mlp
                self.mixers[f"blocks.{l}.mlp.hook_post"] = Mixer((self.cfg.d_mlp), type="neuron", hparams=self.hparams.mixer_hparams)
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
