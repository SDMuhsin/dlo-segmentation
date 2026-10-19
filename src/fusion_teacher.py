#!/usr/bin/env python
"""Inference-time logit-fusion ensemble "P3W (+) QN" (R2-W, QN-backbone 3-way campaign).

P3W  = 3-class SegFormer-B5 (bg / wire / connector): good recall + connector, but paints wire and
       connector on skin / hands on the real videos.
QN   = 2-class SegFormer-B5 (bg / DLO=wire; labels plugs as BACKGROUND): precise, low recall.

fuse(p3w_probs, qn_probs, cfg) combines their softmax outputs into a 3-class distribution:

  wire rule (cfg["wire"]):
    "veto"  : P3W wire mass is kept only where QN P(wire) > cfg["t_low"]; vetoed wire mass -> bg.
              cfg["rescue"]=r>0: P3W wire with P3W P(wire) >= r is kept regardless of QN (rescue).
    "gm"    : geometric mean.  w = P3W_w^(1-b) * QN_w^b ;  o = (P3W_bg+P3W_c)^(1-b) * QN_bg^b  (b=cfg["beta"],
              default .5), renormalised w+o=1; o is split into bg/conn with P3W's bg:conn ratio.
    "none"  : P3W unchanged.
  connector vote (cfg["conn_ref_t"] = t > 0): connector kept only where a second 3-way model
    (cfg["conn_ref"], default A1 ep30 = QN-lineage, no fingertip FP) gives P(conn) >= t; else -> bg.
  connector rule (cfg["conn_gate_px"] = k > 0):
    a connector blob (8-conn component of the fused argmax==2) is kept only if it lies within k px of
    a fused-wire component (plugs terminate cables; fingertips usually don't). Gated blobs: connector
    mass -> bg. (QN says bg for both plugs and fingertips, so geometry is the only connector signal.)

pseudo_label(p3w_probs, qn_probs, cfg) -> uint8 labels with 255 = ignore (see R2W.md for rationale).

FusionModel(nn.Module): forward(x[B,3,H,W] normalised RGB) -> log(fused probs) [B,3,H,W], so every
argmax-based consumer (src/eval_qn3way_suite.py, src/make_connector_sheet.py, infer_video_rgb_only)
treats it as a 3-class model. Specs:  "fusion:<preset>" or "fusion:wire=veto,t_low=0.05,conn_gate_px=15"
(optional "p3w=<ckpt>" / "qn=<ckpt>" overrides inside the same comma list).
Wrappers (no edits to existing code): src/eval_fusion_suite.py, src/make_fusion_sheet.py,
src/infer_video_fusion.py.
"""
import os
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "hf_cache"))

P3W_CKPT = "results/realism_campaign/p3w_lovasz/lovasz050_zoom/best_model.pth"
QN_CKPT = "results/qn_backbone_3way/recon/qn_best_model_hf4.pth"
A1_CKPT = "results/qn_backbone_3way/a1_kdpos/epoch_30.pth"  # default connector-vote reference
EPS = 1e-6

PRESETS = {
    "veto05": dict(wire="veto", t_low=0.05),
    "veto20": dict(wire="veto", t_low=0.20),
    "gm": dict(wire="gm", beta=0.5),
    "veto05_cg15": dict(wire="veto", t_low=0.05, conn_gate_px=15),
    "gm_cg15": dict(wire="gm", beta=0.5, conn_gate_px=15),
    "p3w_cg15": dict(wire="none", conn_gate_px=15),
    "veto05_r95": dict(wire="veto", t_low=0.05, rescue=0.95),
    "veto05_r95_cg40": dict(wire="veto", t_low=0.05, rescue=0.95, conn_gate_px=40),
    "veto05_cvA1": dict(wire="veto", t_low=0.05, conn_ref_t=0.2),
    "veto05_r95_cvA1": dict(wire="veto", t_low=0.05, rescue=0.95, conn_ref_t=0.2),
}
DEFAULT_CFG = dict(wire="veto", t_low=0.05, beta=0.5, conn_gate_px=0, rescue=0.0, conn_ref_t=0.0)


def parse_cfg(s):
    """'veto05' | 'wire=veto,t_low=0.05,conn_gate_px=15[,p3w=..][,qn=..]' -> dict."""
    cfg = dict(DEFAULT_CFG)
    for part in [p for p in s.split(",") if p]:
        if "=" not in part:
            cfg.update(PRESETS[part])
            continue
        k, v = part.split("=", 1)
        if k in ("wire", "p3w", "qn", "conn_ref"):
            cfg[k] = v
        elif k == "conn_gate_px":
            cfg[k] = int(v)
        else:
            cfg[k] = float(v)
    return cfg


def _gate_connector(lab, k):
    """lab: [H,W] uint8 numpy (fused argmax). Returns bool mask of connector px to DROP."""
    con = (lab == 2).astype(np.uint8)
    drop = np.zeros(lab.shape, bool)
    if not con.any():
        return drop
    wire = (lab == 1).astype(np.uint8)
    near = cv2.dilate(wire, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0 \
        if wire.any() else np.zeros(lab.shape, bool)
    n, cc = cv2.connectedComponents(con, connectivity=8)
    if n <= 1:
        return drop
    hit = np.zeros(n, bool)
    hit[np.unique(cc[near & (cc > 0)])] = True
    hit[0] = True
    return ~hit[cc]


@torch.no_grad()
def fuse(p3w_probs, qn_probs, cfg, ref_probs=None):
    """p3w_probs [3,H,W] or [B,3,H,W]; qn_probs [2,H,W] or [B,2,H,W] (softmax, any device).
    Returns (probs [B?,3,H,W] float32, labels [B?,H,W] uint8)."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    single = p3w_probs.dim() == 3
    p = (p3w_probs[None] if single else p3w_probs).float()
    q = (qn_probs[None] if single else qn_probs).float()
    bg, w, c = p[:, 0], p[:, 1], p[:, 2]
    qw = q[:, 1]
    if cfg["wire"] == "veto":
        keep = (qw > cfg["t_low"])
        if cfg.get("rescue", 0) > 0:  # P3W-confident wire survives the veto (QN misses hand-held/blurred cable)
            keep = keep | (w >= cfg["rescue"])
        keep = keep.float()
        out = torch.stack([bg + w * (1 - keep), w * keep, c], 1)
    elif cfg["wire"] == "gm":
        b = cfg["beta"]
        fw = w.clamp_min(EPS) ** (1 - b) * qw.clamp_min(EPS) ** b
        fo = (bg + c).clamp_min(EPS) ** (1 - b) * q[:, 0].clamp_min(EPS) ** b
        z = fw + fo
        fw, fo = fw / z, fo / z
        share_c = c / (bg + c).clamp_min(EPS)
        out = torch.stack([fo * (1 - share_c), fw, fo * share_c], 1)
    elif cfg["wire"] == "none":
        out = p.clone()
    else:
        raise ValueError(cfg["wire"])
    if cfg.get("conn_ref_t", 0) > 0:  # connector vote: keep only where a 2nd 3-way model (A1) agrees
        assert ref_probs is not None, "conn_ref_t needs ref_probs (cfg['conn_ref'] model)"
        r = (ref_probs[None] if single else ref_probs).float()
        okc = (r[:, 2] >= cfg["conn_ref_t"]).float()
        out = torch.stack([out[:, 0] + out[:, 2] * (1 - okc), out[:, 1], out[:, 2] * okc], 1)
    lab = out.argmax(1)
    k = int(cfg.get("conn_gate_px", 0))
    if k > 0:
        lab_np = lab.to(torch.uint8).cpu().numpy()
        drop = torch.from_numpy(np.stack([_gate_connector(l, k) for l in lab_np])).to(out.device)
        if drop.any():
            df = drop.float()
            out = torch.stack([out[:, 0] + out[:, 2] * df, out[:, 1], out[:, 2] * (1 - df)], 1)
            lab = out.argmax(1)
    lab = lab.to(torch.uint8)
    return (out[0], lab[0]) if single else (out, lab)


@torch.no_grad()
def pseudo_label(p3w_probs, qn_probs, cfg, conf=0.7, qn_wire_hi=0.5, ignore_ungated_conn=True, ref_probs=None):
    """Dual-teacher pseudo-label with ignore=255 (recommendation in results/qn_backbone_3way/r2w/R2W.md).
      * fused label where the fused max-prob >= conf, else 255;
      * DISAGREEMENT -> 255: P3W says wire but the rule vetoed it (QN low) -- may be hand-held cable QN
        misses OR skin FP; neither teacher can be trusted, so do not teach bg there;
        QN says wire (>= qn_wire_hi) but fused is not wire -- QN-only recall;
      * connector blobs dropped by the adjacency gate -> 255 (isolated plug vs fingertip ambiguous).
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    single = p3w_probs.dim() == 3
    p = (p3w_probs[None] if single else p3w_probs).float()
    q = (qn_probs[None] if single else qn_probs).float()
    probs, lab = fuse(p, q, cfg, ref_probs)
    ign = probs.max(1).values < conf
    p3_lab = p.argmax(1)
    ign |= (p3_lab == 1) & (lab != 1)
    ign |= (q[:, 1] >= qn_wire_hi) & (lab != 1)
    if ignore_ungated_conn and (int(cfg.get("conn_gate_px", 0)) > 0 or cfg.get("conn_ref_t", 0) > 0):
        ungated = dict(cfg)
        ungated["conn_gate_px"] = 0
        ungated["conn_ref_t"] = 0.0
        _, lab0 = fuse(p, q, ungated)
        ign |= (lab0 == 2) & (lab != 2)
    out = lab.clone()
    out[ign] = 255
    return out[0] if single else out


def _load_seg(ckpt, device):
    from train_rgb_only_sota import SegFormerSegmenter
    st = torch.load(os.path.join(ROOT, ckpt), map_location="cpu", weights_only=False)
    sd = st.get("model_state_dict", st)
    sd = {k.replace("module.", ""): v for k, v in sd.items() if k != "criterion.weight"}
    nc = int(sd["model.decode_head.classifier.weight"].shape[0])
    m = SegFormerSegmenter(backbone_name="nvidia/mit-b5", num_classes=nc, criterion=None)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{ckpt}: missing={missing[:4]} unexpected={unexpected[:4]}")
    return m.eval().to(device), nc


class FusionModel(nn.Module):
    """Two SegFormers -> log fused probs [B,3,H,W] (argmax-compatible 3-class 'logits')."""

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = {**DEFAULT_CFG, **cfg}
        self.p3w, nc1 = _load_seg(self.cfg.get("p3w", P3W_CKPT), device)
        self.qn, nc2 = _load_seg(self.cfg.get("qn", QN_CKPT), device)
        assert (nc1, nc2) == (3, 2), (nc1, nc2)
        self.ref = None
        if self.cfg.get("conn_ref_t", 0) > 0:
            self.ref, nc3 = _load_seg(self.cfg.get("conn_ref", A1_CKPT), device)
            assert nc3 == 3
        # infer_video_rgb_only._infer_num_classes reads model.model.config.num_labels
        object.__setattr__(self, "model", types.SimpleNamespace(config=types.SimpleNamespace(num_labels=3)))

    @torch.no_grad()
    def forward(self, x, label=None):
        lp = self.p3w(x).float().softmax(1)
        lq = self.qn(x).float().softmax(1)
        lr = self.ref(x).float().softmax(1) if self.ref is not None else None
        with torch.autocast("cuda", enabled=False):
            probs, _ = fuse(lp, lq, self.cfg, lr)
        return probs.clamp_min(1e-8).log()


def is_fusion_spec(s):
    return isinstance(s, str) and s.startswith("fusion:")


def load_fusion(spec, device):
    return FusionModel(parse_cfg(spec.split(":", 1)[1]), device).eval()


if __name__ == "__main__":  # unit sanity checks on random tensors
    torch.manual_seed(0)
    p = torch.rand(2, 3, 40, 50).softmax(1)
    q = torch.rand(2, 2, 40, 50).softmax(1)
    for name, c in [(n, c) for n, c in PRESETS.items() if not c.get("conn_ref_t")]:
        pr, lb = fuse(p, q, c)
        assert torch.allclose(pr.sum(1), torch.ones_like(pr[:, 0]), atol=1e-5), name
        print(name, "label hist", np.bincount(lb.flatten().numpy(), minlength=3), "ok")
    pl = pseudo_label(p, q, PRESETS["veto05_cg15"])
    print("pseudo-label ignore frac", float((pl == 255).float().mean()))
