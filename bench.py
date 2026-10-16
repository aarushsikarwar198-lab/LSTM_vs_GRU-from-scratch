import argparse
import json
import os
import random
import time
import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader

from models import Classifier
from train import evaluate, train_one_epoch

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
CFG = dict(
    seeds=[0 ,1, 2 ,3 ],          # use [0] for a pilot run
    lr=1e-3,
    embed_dim=128,
    hidden_size=128,
    dropout=0.5,
    epochs=15,                # set just above where your pilot's val_loss bottoms out
    patience=2,
    min_delta=1e-4,
    grad_clip=1.0,
    deterministic=False,      # True only for the final reported run
    match_window=(128, 200),  # hidden-size search range for the param-matched GRU
    include_same_h=True,      # False drops the "GRU (same h)" condition
    data_path="imdb_preprocessed.pt",
    out_path="results.json",
)

# Optional CLI overrides: python run_experiment.py --seeds 0 --deterministic --out results_final.json
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+")
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--out")
args = parser.parse_args()
if args.seeds:
    CFG["seeds"] = args.seeds
if args.deterministic:
    CFG["deterministic"] = True
if args.out:
    CFG["out_path"] = args.out

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.BCEWithLogitsLoss()

if CFG["deterministic"]:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

data = torch.load(CFG["data_path"], weights_only=False)
vocab_size = data["vocab_size"]
pad_idx = data["pad_idx"]
base_train, val_loader, test_loader = data["train_loader"], data["val_loader"], data["test_loader"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


def build(cell_type, hidden):
    return Classifier(vocab_size, CFG["embed_dim"], hidden, cell_type, CFG["dropout"], pad_idx)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_hidden(cell_type, target_params):
    """Hidden size within CFG['match_window'] whose param count is closest to target."""
    lo, hi = CFG["match_window"]
    best_h, best_gap = None, float("inf")
    for h in range(lo, hi + 1):
        gap = abs(count_params(build(cell_type, h)) - target_params)
        if gap < best_gap:
            best_h, best_gap = h, gap
    if best_h in (lo, hi):
        print(f"WARNING: matched hidden size {best_h} is at the edge of the window "
              f"{(lo, hi)}; widen CFG['match_window'] and rerun.")
    return best_h


def seeded_train_loader(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        base_train.dataset,
        batch_size=base_train.batch_size,
        shuffle=True,
        generator=g,
        collate_fn=base_train.collate_fn,
        drop_last=base_train.drop_last,
    )


def add_grad_clipping(model, optimizer, max_norm):
    orig_step = optimizer.step

    def step(*a, **kw):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return orig_step(*a, **kw)

    optimizer.step = step


# ----------------------------------------------------------------------------
# One training run
# ----------------------------------------------------------------------------
def run(name, cell_type, hidden, seed):
    set_seed(seed)
    t_start = time.time()

    model = build(cell_type, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    if CFG["grad_clip"]:
        add_grad_clipping(model, optimizer, CFG["grad_clip"])
    train_loader = seeded_train_loader(seed)
    n_params = count_params(model)
    n_train = len(train_loader.dataset)

    hist = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "time": []}
    best_val_loss, best_val_acc, best_epoch, bad = float("inf"), 0.0, -1, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for ep in range(CFG["epochs"]):
        sync()
        t0 = time.time()
        loss, acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        sync()
        dt = time.time() - t0
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        for k, v in zip(hist, (loss, acc, val_loss, val_acc, dt)):
            hist[k].append(float(v))
        print(f"[{name} s{seed}] ep {ep+1:02d} | loss {loss:.4f} acc {acc:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | {dt:.1f}s")

        if val_loss < best_val_loss - CFG["min_delta"]:
            best_val_loss, best_val_acc, best_epoch, bad = val_loss, val_acc, ep + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG["patience"]:
                print(f"[{name} s{seed}] early stop at epoch {ep+1}")
                break

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

    med_epoch = float(np.median(hist["time"]))
    return {
        "cell": cell_type,
        "hidden": hidden,
        "seed": seed,
        "params": n_params,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "median_s_per_epoch": med_epoch,
        "samples_per_s": n_train / med_epoch,
        "peak_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0.0,
        "total_time": time.time() - t_start,
        "history": hist,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    h = CFG["hidden_size"]
    lstm_params = count_params(build("LSTM", h))
    gru_h_matched = match_hidden("GRU", lstm_params)

    conditions = {"LSTM": ("LSTM", h)}
    if CFG["include_same_h"]:
        conditions["GRU (same h)"] = ("GRU", h)
    conditions["GRU (same #par)"] = ("GRU", gru_h_matched)

    print(f"device={device} | deterministic={CFG['deterministic']} | "
          f"LSTM h={h} ({lstm_params:,} params) | matched GRU h={gru_h_matched}")

    # resume: load earlier results and skip finished (condition, seed) pairs.
    # Delete the JSON file if you change any hyperparameter, or old and new runs get mixed.
    results = {name: [] for name in conditions}
    if os.path.exists(CFG["out_path"]):
        with open(CFG["out_path"]) as f:
            old = json.load(f).get("results", {})
        for name in conditions:
            results[name] = old.get(name, [])

    for seed in CFG["seeds"]:
        for name, (cell, hidden) in conditions.items():
            if any(r["seed"] == seed for r in results[name]):
                print(f"skip {name} seed {seed} (already done)")
                continue
            results[name].append(run(name, cell, hidden, seed))
            with open(CFG["out_path"], "w") as f:
                json.dump({"config": CFG, "results": results}, f, indent=2, default=float)

    # ---- summary table (mean ± std over seeds) ----
    metrics = ["params", "best_epoch", "best_val_acc", "test_acc", "test_loss",
               "median_s_per_epoch", "samples_per_s", "peak_mem_mb"]
    names = list(conditions)
    print(f"\n{'metric':<22}" + "".join(f"{n:<26}" for n in names))
    for k in metrics:
        row = f"{k:<22}"
        for n in names:
            v = np.array([r[k] for r in results[n]], dtype=float)
            sd = v.std(ddof=1) if len(v) > 1 else float("nan")
            row += f"{v.mean():.4f} ± {sd:.4f}".ljust(26)
        print(row)