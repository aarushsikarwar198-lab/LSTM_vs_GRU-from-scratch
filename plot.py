import json
import numpy as np
import matplotlib.pyplot as plt

with open("results.json") as f:
    R = json.load(f)["results"]

groups = list(R)
col = {g: plt.cm.tab10.colors[i] for i, g in enumerate(groups)}

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax = ax.ravel()

# per-seed curves
for g, runs in R.items():
    for r in runs:
        h = r["history"]
        ep = range(1, len(h["val_acc"]) + 1)
        ax[0].plot(ep, h["val_acc"], color=col[g], alpha=0.6)
        ax[1].plot(ep, h["val_loss"], color=col[g], alpha=0.6)
    ax[0].plot([], [], color=col[g], label=g)

# bars: test acc and median s/epoch (mean ± std across seeds)
for i, g in enumerate(groups):
    acc = [r["test_acc"] for r in R[g]]
    t = [r["median_s_per_epoch"] for r in R[g]]
    ax[2].bar(g, np.mean(acc), yerr=np.std(acc), color=col[g], capsize=4)
    ax[3].bar(g, np.mean(t), yerr=np.std(t), color=col[g], capsize=4)

ax[0].set_title("Val accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].set_title("Val loss");     ax[1].set_xlabel("epoch")
ax[2].set_title("Test accuracy (mean ± std, 4 seeds)")
ax[2].set_ylim(0.80, 0.88)
ax[3].set_title("Median s/epoch (mean ± std)"); ax[3].set_ylabel("seconds")
for a in ax[2:]:
    a.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("comparison.png", dpi=150)
plt.show()