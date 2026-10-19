#!/usr/bin/env python
"""Run src/eval_qn3way_suite.py with fusion specs (src/fusion_teacher.py) accepted as CKPT.

  ./env/bin/python src/eval_fusion_suite.py fus_veto05=fusion:veto05 fus_gm=fusion:gm --no-probes
Regular NAME=CKPT specs pass through unchanged. Probes are ckpt-path based -> use --no-probes for fusion.
"""
import sys
import eval_qn3way_suite as S  # sets CUDA_VISIBLE_DEVICES from --gpu (default 1) at import
import fusion_teacher as FT

_orig_load = S.load


def load(ckpt):
    if FT.is_fusion_spec(ckpt):
        return FT.load_fusion(ckpt, S.DEV), 3
    return _orig_load(ckpt)


S.load = load
if __name__ == "__main__":
    S.main()
