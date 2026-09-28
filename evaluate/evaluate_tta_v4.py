"""evaluate_tta.py for Stage-2 v4 checkpoints (Markov label-refine layer).

evaluate_tta.py builds a plain ConvNeXt2 and loads with strict=False, which would silently
drop a v4 checkpoint's `label_refine.*` weights and score the un-refined logits. This wrapper
swaps in train/markov_layer.load_convnext2_for_state_dict (ConvNeXt2Markov when the checkpoint
has the layer, plain ConvNeXt2 otherwise) and otherwise runs evaluate_tta.py unchanged - same
CLI, same TTA / ensembling / --dump-probs behaviour, so v3 and v4 checkpoints can be mixed.

    python evaluate_tta_v4.py --dump-probs DIR --out-json OUT.json ckptA@768 ckptB@1024 ...
"""

import torch

import evaluate_tta as base
from markov_layer import load_convnext2_for_state_dict


def load_model(ckpt_path, device):
    sd = torch.load(ckpt_path, map_location="cpu")
    model, res = load_convnext2_for_state_dict(sd)
    kind = "ConvNeXt2Markov" if hasattr(model, "label_refine") else "ConvNeXt2"
    base.logger.info(f"loaded {ckpt_path} as {kind} "
                     f"(missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)})")
    return model.to(device).eval()


base.load_model = load_model

if __name__ == "__main__":
    base.main()
