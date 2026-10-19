#!/usr/bin/env python
"""src/make_connector_sheet.py with fusion specs (src/fusion_teacher.py) accepted as CKPT.
Same CLI:  ./env/bin/python src/make_fusion_sheet.py p3w=... qn=... fus=fusion:veto05_cg15 --out ...
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_connector_sheet as M  # parses sys.argv (same CLI) and sets CUDA_VISIBLE_DEVICES
import fusion_teacher as FT

_load, _ckey = M.load, M.ckey


def load(ckpt):
    return FT.load_fusion(ckpt, M.DEV) if FT.is_fusion_spec(ckpt) else _load(ckpt)


def ckey(ckpt, frames):
    if not FT.is_fusion_spec(ckpt):
        return _ckey(ckpt, frames)
    cfg = FT.parse_cfg(ckpt.split(":", 1)[1])
    parts = [ckpt] + [_ckey(cfg.get(k, d), frames) for k, d in (("p3w", FT.P3W_CKPT), ("qn", FT.QN_CKPT), ("conn_ref", FT.A1_CKPT))]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


M.load, M.ckey = load, ckey
if __name__ == "__main__":
    M.main()
