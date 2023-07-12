# %%
from itertools import islice
from typing import Any, Dict

import torch

from mixer import MixerHparams
from mixing_transformer import MixingTransformer, MixingTransformerHparams
from utils import DEVICE, get_owt_dataset, base_model, tokenizer

# %%
# mix_model = MixingTransformer(base_model)

dataset = get_owt_dataset()

# %%


def get_loss(model, max_steps=10, batch_size=4):
    losses = []
    batch_iter = dataset.iter(batch_size)
    for step in range(max_steps):
        ref_toks = torch.stack(next(batch_iter)["toks"])
        alt_toks = torch.stack(next(batch_iter)["toks"])
        out = model.forward(ref_toks, alt_toks)

        loss = torch.nn.CrossEntropyLoss(reduction="mean")(out[:, :-1].flatten(0, 1), ref_toks[:, 1:].flatten())
        losses.append(loss.item())

    return sum(losses) / len(losses)


# %%

print("losses for different p")
for p in torch.arange(0, 1.01, 0.1):
    hparams = MixingTransformerHparams(mixer_hparams=MixerHparams(init_p=p))
    model = MixingTransformer(base_model, hparams=hparams)
    with torch.no_grad():
        loss = get_loss(model)

    print(f"p={p:.2f} | loss={loss:.2f}")


# %%


##### Misc testing


# batch = ["The quick brown fox jumped over the lazy dog", "The dog didn't care much. It was used to it."]
# toks = base_model.to_tokens(batch)

# out = mix_model.forward(toks[[0]], toks[[1]])

# %%

# print("Mixed (ref) top predicitons:")
# for i in range(10):
#     top_tok_id = out[0, i].argmax(-1)
#     print(
#         base_model.to_str_tokens(batch[0])[i],
#         repr(base_model.to_string(top_tok_id.item())),
#         out[0, i, top_tok_id].item(),
#         sep="\t| ",
#     )
