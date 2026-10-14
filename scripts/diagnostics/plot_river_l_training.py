import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = [11, 42, 101, 202, 314]
BASE = "artifacts/river_l"
OUT = "artifacts/figures"
os.makedirs(OUT, exist_ok=True)

histories = {}
for seed in SEEDS:
    path = f"{BASE}/river_l_seed{seed}.manifest.json"
    with open(path) as f:
        m = json.load(f)
    histories[seed] = m["history"]

metrics = [
    ("loss", "Total loss"),
    ("revision_loss", "Revision loss"),
    ("invariance_loss", "Invariance loss"),
    ("update_loss", "Update loss"),
    ("calibration_loss", "Calibration loss"),
    ("state_accuracy", "State accuracy"),
    ("selected_accuracy", "Selected-hypothesis accuracy"),
]

colors = plt.cm.tab10.colors

# Individual metric plots, all seeds overlaid
for key, title in metrics:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, seed in enumerate(SEEDS):
        h = histories[seed]
        epochs = [row["epoch"] for row in h]
        vals = [row.get(key) for row in h]
        ax.plot(epochs, vals, label=f"seed {seed}", color=colors[i % len(colors)])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.set_title(f"RIVER-L training: {title} (all seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/river_l_training_{key}.png", dpi=150)
    plt.close(fig)

# Combined dashboard: 2x2 grid of the four loss components + accuracy
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
dashboard_metrics = [
    ("loss", "Total loss"),
    ("revision_loss", "Revision loss"),
    ("state_accuracy", "State accuracy"),
    ("selected_accuracy", "Selected-hypothesis accuracy"),
]
for ax, (key, title) in zip(axes.flat, dashboard_metrics):
    for i, seed in enumerate(SEEDS):
        h = histories[seed]
        epochs = [row["epoch"] for row in h]
        vals = [row.get(key) for row in h]
        ax.plot(epochs, vals, label=f"seed {seed}", color=colors[i % len(colors)])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(alpha=0.3)
axes.flat[0].legend(fontsize=8)
fig.suptitle("RIVER-L training dashboard (5 seeds)")
fig.tight_layout()
fig.savefig(f"{OUT}/river_l_training_dashboard.png", dpi=150)
plt.close(fig)

# Summary table of best validation loss / epochs per seed
summary = []
for seed in SEEDS:
    path = f"{BASE}/river_l_seed{seed}.manifest.json"
    with open(path) as f:
        m = json.load(f)
    summary.append({
        "seed": seed,
        "epochs_ran": m["epochs_ran"],
        "best_validation_loss": m["best_validation_loss"],
    })

with open(f"{OUT}/river_l_training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

for row in summary:
    print(row)

print(f"\nWrote plots to {OUT}/")
