from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from utils import DEVICE


@dataclass(frozen=True)
class MixerHparams:
    init_p: float = 0.9
    init_q: float = 0.1
    adversarial: bool = False


class Mixer(nn.Module):
    def __init__(
        self,
        shape: Tuple[int],
        hparams: MixerHparams = MixerHparams(),
        type: Optional[str] = None,  # non-functional, but a helpful tag
    ):
        """
        shape should allow broadcasting with activations. eg:
            activations: [batch, seqpos, heads, embsize]
            weights: [heads, 1]
        """
        super().__init__()
        self.adversarial = hparams.adversarial
        self.p_logit = self.init_param(hparams.init_p, shape)
        if self.adversarial:
            self.q_logit = self.init_param(hparams.init_q, shape)

    @staticmethod
    def init_param(p: float, shape: Tuple[int]):
        assert p >= 0 and p <= 1
        logit = torch.special.logit(torch.as_tensor(p), eps=1e-6)
        return nn.Parameter(torch.full(fill_value=logit, size=shape, device=DEVICE))

    @property
    def p(self):
        return torch.sigmoid(self.p_logit)

    @property
    def q(self):
        return torch.sigmoid(self.q_logit)

    @property
    def w(self) -> torch.Tensor:
        """
        This is the total mix proportion, including both protagonist and adversarial contributions. Will be a tensor with values in [0,1].
        """
        return self.p + self.q - self.p * self.q if self.adversarial else self.p

    def forward(self, x, y):
        return x * self.w + y * (1 - self.w)
