import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CSV_PATH = os.path.join("checkpoints", "metrics.csv")
OUT_PATH = os.path.join("checkpoints", "metrics.png")

df = pd.read_csv(CSV_PATH)

sns.set_theme(style="whitegrid")
palette = sns.color_palette("muted")

fig, ax_loss = plt.subplots(figsize=(9, 5))
ax_ppl = ax_loss.twinx()

# Plot only loss lines — perplexity = exp(loss), shown as a secondary scale on the right
sns.lineplot(data=df, x="epoch", y="train_loss", ax=ax_loss,
             color=palette[0], marker="o", label="Train")
sns.lineplot(data=df, x="epoch", y="val_loss", ax=ax_loss,
             color=palette[1], marker="o", label="Val")

# Align right axis as exp() of the left axis so the gridlines stay consistent
y_min, y_max = ax_loss.get_ylim()
ax_ppl.set_ylim(np.exp(y_min), np.exp(y_max))

# Set right-axis ticks at the same positions (in perplexity space) as the left ticks
loss_ticks = ax_loss.get_yticks()
ppl_ticks = np.exp(loss_ticks)
ax_ppl.set_yticks(ppl_ticks)
ax_ppl.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Cross-Entropy Loss")
ax_ppl.set_ylabel("Perplexity")
ax_loss.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax_loss.legend(loc="upper right")

plt.title("Train vs. Validation — Loss & Perplexity")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
