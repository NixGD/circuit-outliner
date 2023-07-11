from typing import Optional, Tuple

import torch

from utils import DEVICE


class Mixer:
    def __init__(
        self,
        weight_shape: Tuple[int],
        init_p: float = 0.9,
        init_q: float = 0.1,
        adversarial: bool = False,
        type: Optional[str] = None, # for neater sorting, etc
    ):
        """
        weight_shape should allow broadcasting with activations. eg:
        activations: [batch, seqpos, heads, embsize]
        weights: [heads, 1]
        """
        self.adversarial = adversarial
        self.p = torch.full(weight_shape, fill_value=init_p, device=DEVICE)
        if adversarial:
            self.q = torch.full(weight_shape, fill_value=init_q, device=DEVICE)

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
