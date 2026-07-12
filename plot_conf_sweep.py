import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("conf_sweep.csv")
df["Precision"] = df["Precision"].astype(float)
df["Recall"]    = df["Recall"].astype(float)
df["F1"]        = df["F1"].astype(float)

# Class == "A" rows are already the macro avg (Moose + Deer) per fold.
# Average those across all 20 CV folds per confidence threshold.
avg = (
    df[df["Class"] == "A"]
    .groupby("Conf Threshold")[["Precision", "Recall", "F1"]]
    .mean()
    .reset_index()
    .sort_values("Conf Threshold")
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(avg["Conf Threshold"], avg["Precision"],
        color="steelblue",  marker="o", markersize=4, linewidth=2, label="Precision")
ax.plot(avg["Conf Threshold"], avg["Recall"],
        color="darkorange", marker="o", markersize=4, linewidth=2, label="Recall")
ax.plot(avg["Conf Threshold"], avg["F1"],
        color="seagreen",   marker="o", markersize=4, linewidth=2, label="F1")

ax.set_xlabel("Confidence Threshold", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title(
    "Precision / Recall / F1 vs Confidence Threshold\n"
    "(macro avg Moose & Deer, mean over 20 CV folds)",
    fontsize=11,
)
ax.set_xlim(0.03, 0.97)
ax.set_ylim(0, 1.05)
ax.set_xticks(avg["Conf Threshold"])
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("conf_sweep_plot.png", dpi=150)
plt.show()
print("Saved conf_sweep_plot.png")

print("\nThreshold averages (macro avg over 20 folds):")
print(avg.to_string(index=False, float_format="%.4f"))