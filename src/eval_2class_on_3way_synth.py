#!/usr/bin/env python
"""Score a 2-class (bg/DLO) or 3-class (bg/wire/connector) SegFormer on the 3-way synthetic
val set, COLLAPSED to binary DLO (GT wire + connector -> DLO), and report how GT-connector
pixels are labelled (does the model treat connectors as DLO or background?).

Also measures inference throughput + peak GPU memory (batch 1 and batch 8, fp32 and AMP-fp16).

Checkpoints must use transformers-v4 key names (convert v5 ones with
src/convert_hf5_segformer_ckpt.py first; this script refuses a checkpoint with missing keys).

Usage:
  CUDA_VISIBLE_DEVICES=1 ./env/bin/python src/eval_2class_on_3way_synth.py \
      --ckpt NAME=path/to/best_model.pth [--ckpt NAME2=...] \
      --out results/qn_backbone_3way/recon/synth_collapsed.json
"""
import argparse, json, os, sys, time
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "data", "hf_cache"))
from train_rgb_only_sota import SegFormerSegmenter  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(ckpt, device):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = st.get("model_state_dict", st)
    nc = int(sd["model.decode_head.classifier.weight"].shape[0])
    m = SegFormerSegmenter(backbone_name="nvidia/mit-b5", num_classes=nc, criterion=None)
    sd = {k: v for k, v in sd.items() if k != "criterion.weight"}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{ckpt}: missing={missing[:4]} unexpected={unexpected[:4]} "
                         "(v5 checkpoint? run convert_hf5_segformer_ckpt.py)")
    return m.eval().to(device), nc


def batches(data, names, bs):
    for i in range(0, len(names), bs):
        rgbs, labs = [], []
        for n in names[i:i + bs]:
            bgr = cv2.imread(os.path.join(data, "RGB", n + ".png"))
            rgbs.append(bgr[:, :, ::-1].copy())
            labs.append(cv2.imread(os.path.join(data, "Label", n + ".png"), -1))
        x = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float() / 255.0
        yield (x - MEAN) / STD, np.stack(labs)


@torch.no_grad()
def evaluate(m, nc, data, names, device, bs, dlo_vals=(1, 2), con_val=2, wire_val=1):
    c = dict(tp=0, fp=0, fn=0, n_con=0, con_dlo=0, n_wire=0, wire_dlo=0,
             con_as_wire=0, con_as_con=0, wire_as_con=0)
    for x, lab in batches(data, names, bs):
        with torch.autocast("cuda", dtype=torch.float16):
            pred = m(x.to(device)).argmax(1).cpu().numpy()
        pd = (pred >= 1)                          # DLO for 2-class; wire|connector for 3-class
        gd = np.isin(lab, dlo_vals)
        valid = lab != 255
        c["tp"] += int((pd & gd & valid).sum())
        c["fp"] += int((pd & ~gd & valid).sum())
        c["fn"] += int((~pd & gd & valid).sum())
        gc, gw = lab == con_val, lab == wire_val
        c["n_con"] += int(gc.sum()); c["con_dlo"] += int((pd & gc).sum())
        c["n_wire"] += int(gw.sum()); c["wire_dlo"] += int((pd & gw).sum())
        if nc == 3:
            c["con_as_wire"] += int(((pred == 1) & gc).sum())
            c["con_as_con"] += int(((pred == 2) & gc).sum())
            c["wire_as_con"] += int(((pred == 2) & gw).sum())
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    r = {"num_classes": nc, "n_images": len(names),
         "iou_dlo_collapsed": tp / max(tp + fp + fn, 1),
         "precision_dlo": tp / max(tp + fp, 1), "recall_dlo": tp / max(tp + fn, 1),
         "gt_connector_px": c["n_con"],
         "gt_connector_pred_dlo_frac": c["con_dlo"] / max(c["n_con"], 1),
         "gt_connector_pred_bg_frac": 1 - c["con_dlo"] / max(c["n_con"], 1),
         "gt_wire_pred_dlo_frac": c["wire_dlo"] / max(c["n_wire"], 1)}
    if nc == 3:
        r["gt_connector_pred_wire_frac"] = c["con_as_wire"] / max(c["n_con"], 1)
        r["gt_connector_pred_connector_frac"] = c["con_as_con"] / max(c["n_con"], 1)
    return r


@torch.no_grad()
def throughput(m, device, bs, amp, iters=30):
    x = torch.randn(bs, 3, 480, 640, device=device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if amp else torch.no_grad
    for _ in range(5):
        with ctx():
            m(x)
    torch.cuda.synchronize(device)
    t0 = time.time()
    for _ in range(iters):
        with ctx():
            m(x).argmax(1)
    torch.cuda.synchronize(device)
    dt = time.time() - t0
    return {"batch": bs, "amp_fp16": amp, "img_per_s": bs * iters / dt,
            "ms_per_batch": 1000 * dt / iters,
            "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--data", default="data/dformer_dataset_3way_connscale3")
    ap.add_argument("--split", default="test.txt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--no-throughput", action="store_true")
    ap.add_argument("--dlo-values", default="1,2",
                    help="GT label values counted as DLO (MovingCables: 4)")
    ap.add_argument("--con-value", type=int, default=2)
    ap.add_argument("--wire-value", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    names = [os.path.splitext(os.path.basename(l.split()[0]))[0]
             for l in open(os.path.join(a.data, a.split)) if l.strip()]
    out = {}
    for spec in a.ckpt:
        name, _, path = spec.partition("=")
        m, nc = load(path, dev)
        r = evaluate(m, nc, a.data, names, dev, a.bs,
                     tuple(int(v) for v in a.dlo_values.split(",")), a.con_value, a.wire_value)
        if not a.no_throughput:
            r["throughput"] = [throughput(m, dev, b, amp) for b in (1, 8) for amp in (False, True)]
        out[name] = r
        print(name, json.dumps(r, indent=1), flush=True)
        del m; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("->", a.out)


if __name__ == "__main__":
    main()
