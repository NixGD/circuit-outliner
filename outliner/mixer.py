from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from utils import DEVICE


@dataclass(frozen=True)
class MixerHparams:
    init_p: float = 0.9
    init_q: float = 0.1
    adversarial: bool = False


class Mixer:
    def __init__(
        self,
        weight_shape: Tuple[int],
        hparams: MixerHparams = MixerHparams(),
        type: Optional[str] = None,  # non-functional, but a helpful tag
    ):
        """
        weight_shape should allow broadcasting with activations. eg:
        activations: [batch, seqpos, heads, embsize]
        weights: [heads, 1]
        """
        self.adversarial = hparams.adversarial
        self.p = torch.full(weight_shape, fill_value=hparams.init_p, device=DEVICE)
        if self.adversarial:
            self.q = torch.full(weight_shape, fill_value=hparams.init_q, device=DEVICE)

    def get_mix_weights(self):
        if self.adversarial:
            raise self.p + self.q - self.p * self.q
        else:
            return self.p

    def __call__(self, x, y):
        """
        broadcasts, so
        """
        w = self.get_mix_weights()
        return x * w + y * (1 - w)
