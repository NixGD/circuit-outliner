# %%
import torch
from transformer_lens import HookedTransformer

from mixing_transformer import MixingTransformer
from utils import DEVICE

base_model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

# %%
mix_model = MixingTransformer(base_model, mix_heads=True, mix_mlps=False, mix_neurons=False)

# %%

batch = ["The quick brown fox jumped over the lazy dog", "The dog didn't care much. It was used to it."]
toks = base_model.to_tokens(batch)

out = mix_model.forward(toks[[0]], toks[[1]])

# %%

print("Mixed (ref) top predicitons:")
for i in range(10):
    top_tok_id = out[0, i].argmax(-1)
    print(
        base_model.to_str_tokens(batch[0])[i],
        repr(base_model.to_string(top_tok_id.item())),
        out[0, i, top_tok_id].item(),
        sep="\t| ",
    )

print("\n\nUnmixed (alt) top predicitons:")
for i in range(10):
    top_tok_id = out[1, i].argmax(-1)
    print(
        base_model.to_str_tokens(batch[1])[i],
        repr(base_model.to_string(top_tok_id.item())),
        out[1, i, top_tok_id].item(),
        sep="\t| ",
    )


# %%
