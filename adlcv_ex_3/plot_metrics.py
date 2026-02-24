import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_PATH = os.path.join("checkpoints", "metrics.csv")
OUT_PATH = os.path.join("checkpoints", "metrics.png")

df = pd.read_csv(CSV_PATH)

sns.set_theme(style="whitegrid")
palette = sns.color_palette("muted")

fig, ax = plt.subplots(figsize=(7, 4))

sns.lineplot(data=df, x="epoch", y="train_loss", ax=ax,
             color=palette[0], marker="o", label="Train")
sns.lineplot(data=df, x="epoch", y="val_loss", ax=ax,
             color=palette[1], marker="o", label="Val")

best_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
best_val_loss = df["val_loss"].min()
ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
ax.annotate(
    f"best val (epoch {best_epoch})",
    xy=(best_epoch, best_val_loss),
    xytext=(best_epoch + 0.3, best_val_loss + 0.3),
    fontsize=8,
    color="gray",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.legend()

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
