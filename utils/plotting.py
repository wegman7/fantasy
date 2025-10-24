from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd


def scatter_adp_vs_final(df: pd.DataFrame, *, title: Optional[str] = None):
    if df.empty:
        return
    max_axis = max(df["espn_adp"].max(), df["final_rank"].max())
    plt.figure(figsize=(6, 6))
    plt.scatter(df["espn_adp"], df["final_rank"], alpha=0.7)
    plt.plot([0, max_axis], [0, max_axis], "r--", linewidth=1)
    plt.xlabel("ESPN ADP (expected)")
    plt.ylabel("Final Rank (actual)")
    if title:
        plt.title(title)
    else:
        plt.title("ESPN ADP vs Actual Performance")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
