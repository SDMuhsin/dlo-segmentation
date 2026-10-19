#!/usr/bin/env python
"""Convert a SegFormer checkpoint saved under transformers v5 module naming to the
transformers v4 naming this repo's env (transformers 4.57.6) uses.

Why: the Queen's partner checkpoint `data/best_model.pth` (SegFormer mit-b5, 2-class,
trained on ElectricWires with our own trainer) was saved with transformers v5, which
renamed SegFormer modules (`segformer.stages.N.blocks.M.attention.q_proj` instead of
`segformer.encoder.block.N.M.attention.self.query`, `decode_head.linear_projections`
instead of `decode_head.linear_c`, ...). Every loader here uses `strict=False`, so
loading the raw file SILENTLY keeps only the 8 decode-head keys whose names did not
change and leaves the whole encoder at ImageNet init. This script renames the keys
1:1 (architecture and tensor shapes are identical), verifies exact coverage against a
freshly-built v4 model, and writes a checkpoint with the same dict structure
(epoch/metrics/args/model_state_dict/config) that every tool here loads unchanged
(infer_video_rgb_only.py, the probes, train_rgb_only_sota.py --init-checkpoint).

Usage:
  ./env/bin/python src/convert_hf5_segformer_ckpt.py \
      --in data/best_model.pth \
      --out results/qn_backbone_3way/recon/qn_best_model_hf4.pth
"""
import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RULES = [
    # encoder patch embeddings / stage norms
    (r"^model\.segformer\.stages\.(\d+)\.patch_embeddings\.",
     r"model.segformer.encoder.patch_embeddings.\1."),
    (r"^model\.segformer\.stages\.(\d+)\.layer_norm\.",
     r"model.segformer.encoder.layer_norm.\1."),
    # transformer blocks
    (r"^model\.segformer\.stages\.(\d+)\.blocks\.(\d+)\.",
     r"model.segformer.encoder.block.\1.\2."),
]
BLOCK_RULES = [
    (".layernorm_before.", ".layer_norm_1."),
    (".layernorm_after.", ".layer_norm_2."),
    (".attention.q_proj.", ".attention.self.query."),
    (".attention.k_proj.", ".attention.self.key."),
    (".attention.v_proj.", ".attention.self.value."),
    (".attention.o_proj.", ".attention.output.dense."),
    (".attention.sequence_reduction.sequence_reduction.", ".attention.self.sr."),
    (".attention.sequence_reduction.layer_norm.", ".attention.self.layer_norm."),
    (".mlp.fc1.", ".mlp.dense1."),
    (".mlp.fc2.", ".mlp.dense2."),
]


def rename(k):
    for pat, rep in RULES:
        k2 = re.sub(pat, rep, k)
        if k2 != k:
            k = k2
            break
    if ".encoder.block." in k:
        for a, b in BLOCK_RULES:
            k = k.replace(a, b)
    k = k.replace("model.decode_head.linear_projections.", "model.decode_head.linear_c.")
    return k


def is_hf5(sd):
    return any(k.startswith("model.segformer.stages.") for k in sd)


def convert_state_dict(sd):
    out = {}
    for k, v in sd.items():
        nk = rename(k)
        if nk in out:
            raise RuntimeError(f"rename collision: {k} -> {nk}")
        out[nk] = v
    return out


def verify(sd, backbone, num_classes):
    from train_rgb_only_sota import SegFormerSegmenter
    import torch.nn as nn
    crit = nn.CrossEntropyLoss(weight=torch.ones(num_classes), reduction="none")
    m = SegFormerSegmenter(backbone_name=backbone, num_classes=num_classes, criterion=crit)
    tgt = m.state_dict()
    missing = sorted(set(tgt) - set(sd))
    unexpected = sorted(set(sd) - set(tgt))
    bad_shape = [k for k in sd if k in tgt and sd[k].shape != tgt[k].shape]
    return missing, unexpected, bad_shape, len(tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = torch.load(args.inp, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    if not is_hf5(sd):
        print("checkpoint already uses v4 naming; nothing to do")
    new_sd = convert_state_dict(sd)
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    a = ck.get("args", {}) if isinstance(ck, dict) else {}
    backbone = cfg.get("backbone") or a.get("backbone") or "nvidia/mit-b5"
    nc = int(cfg.get("num_classes") or a.get("num_classes") or 2)
    missing, unexpected, bad, n_tgt = verify(new_sd, backbone, nc)
    print(f"keys: in={len(sd)} out={len(new_sd)} target={n_tgt}")
    print(f"missing={missing} unexpected={unexpected} shape_mismatch={bad}")
    if missing or unexpected or bad:
        raise SystemExit("conversion NOT exact; refusing to write")
    out = dict(ck) if isinstance(ck, dict) and "model_state_dict" in ck else {}
    out["model_state_dict"] = new_sd
    out["converted_from"] = os.path.abspath(args.inp)
    out["conversion"] = "transformers-v5 -> v4 SegFormer key names (1:1 rename, tensors unchanged)"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
