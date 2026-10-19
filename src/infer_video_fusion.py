#!/usr/bin/env python
"""src/infer_video_rgb_only.py with --ckpt fusion:<cfg> accepted (src/fusion_teacher.py).
  ./env/bin/python src/infer_video_fusion.py --video data/dlo_real_sample_2.mp4 --ckpt fusion:veto05_cg15 \
      --out-dir ... --device cuda:1 --cls2 connector
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer_video_rgb_only as V
import fusion_teacher as FT

_lm = V.load_model


def load_model(ckpt, device, backbone=None, **kw):
    if FT.is_fusion_spec(ckpt) or FT.is_fusion_spec(os.path.basename(ckpt)):
        return FT.load_fusion(ckpt[ckpt.index("fusion:"):], device)
    return _lm(ckpt, device, **({"backbone": backbone} if backbone else {}))


V.load_model = load_model
if __name__ == "__main__":
    V.main()
