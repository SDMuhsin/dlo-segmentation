#!/usr/bin/env python
"""Run src/eval_real_video_meter.py with fusion specs (src/fusion_teacher.py) accepted as CKPT.
  ./env/bin/python src/eval_fusion_meter.py fus_veto05_cvA1=fusion:veto05_cvA1 [NAME=CKPT ...]
"""
import json
import os
import sys
import time

import eval_real_video_meter as M  # sets CUDA_VISIBLE_DEVICES from --gpu (default 1)
import fusion_teacher as FT
import numpy as np
import torch

_orig_pred_cache = M.pred_cache


def _stat(ck):
    st = os.stat(os.path.join(M.ROOT, ck))
    return [ck, int(st.st_mtime), st.st_size]


def pred_cache(cache, stride, name, ckpt, frames):
    if not FT.is_fusion_spec(ckpt):
        return _orig_pred_cache(cache, stride, name, ckpt, frames)
    cfg = FT.parse_cfg(ckpt.split(":", 1)[1])
    key = {"ckpt": ckpt, "stride": stride, "parts": [_stat(cfg.get(k, d)) for k, d in
           (("p3w", FT.P3W_CKPT), ("qn", FT.QN_CKPT), ("conn_ref", FT.A1_CKPT))]}
    d = os.path.join(cache, "preds")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{name}_s{stride}.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        if json.loads(str(z["key"])) == key:
            return {v: z[v] for v in M.VIDEOS}, int(z["nc"]), 0.0
    t0 = time.time()
    m = FT.load_fusion(ckpt, M.DEV)
    out = {}
    with torch.no_grad():
        for v in M.VIDEOS:
            fr = frames[v][0]
            ps = []
            for i in range(0, len(fr), M.BS):
                x = torch.from_numpy(np.ascontiguousarray(fr[i:i + M.BS][..., ::-1])).permute(0, 3, 1, 2)
                x = (x.float() / 255.0 - M.MEAN) / M.STD
                ps.append(m(x.to(M.DEV)).argmax(1).to(torch.uint8).cpu().numpy())
            out[v] = np.concatenate(ps)
    del m
    torch.cuda.empty_cache()
    dt = time.time() - t0
    np.savez_compressed(p, key=json.dumps(key), nc=3, **out)
    print(f"[preds] {name}: fusion, inference {dt:.0f}s", flush=True)
    return out, 3, dt


M.pred_cache = pred_cache
if __name__ == "__main__":
    M.main()
