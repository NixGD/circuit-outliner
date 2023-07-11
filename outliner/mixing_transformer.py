import torch
from torch import nn
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from mixer import Mixer


class MixingTransformer(nn.Module):
    def __init__(
        self,
        base_model: HookedTransformer,
        mix_heads: bool = True,
        mix_neurons: bool = False,
        mix_mlps: bool = False, # 1 parameter per full mlp layer
    ):
        super().__init__()
        self.base_model = base_model
        self.cfg = self.base_model.cfg
        n_layers, n_heads = self.base_model.cfg.n_layers, self.base_model.cfg.n_heads

        self.mixers = {} # will be in model order
        for l in range(self.cfg.n_layers):
            if mix_heads:
                self.mixers[f"blocks.{l}.attn.hook_z"] = Mixer((self.cfg.n_heads, 1), type="head")
            if mix_neurons:
                assert not mix_mlp
                self.mixers[f"blocks.{l}.mlp.hook_post"] = Mixer((self.cfg.d_mlp), type="neuron")
            if mix_mlps:
                assert not mix_neurons
                self.mixers[f"blocks.{l}.hook_mlp_out"] = Mixer((1,), type="mlp")

    def forward(self, ref_in, alt_in):
        ins = torch.cat([ref_in, alt_in], axis=0)
        ref_idxs = torch.arange(len(ins)) < len(ref_in)

        def hook(x, hook: HookPoint):
            mixer = self.mixers[hook.name]
            x[ref_idxs] = mixer(x[ref_idxs], x[~ref_idxs])
            return x

        is_mixed = lambda name: name in self.mixers
        return self.base_model.run_with_hooks(ins, fwd_hooks=[(is_mixed, hook)])
