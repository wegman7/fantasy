# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# ADP vs ECR analysis driver
#
# This notebook-script compares ESPN ADP and FantasyPros ECR against actual end-of-season ranks
# using data from fp_adp/, fp_ecr/, and fp_season_stats/.

# %%
from itables import init_notebook_mode, show
from IPython.display import display
import pandas as pd
init_notebook_mode(all_interactive=True)

# %% [markdown]
# Parameters

# %%
year = "2024"
scoring = "ppr"  # ppr | half | std
pos = "wr"       # qb | rb | wr | te | dst | k

# %% [markdown]
# Load data

# %%
from utils.paths import fp_stats_path, fp_adp_path, fp_ecr_path
from utils.loaders import read_csv

stats_path = fp_stats_path(year, scoring, pos)
adp_path = fp_adp_path(year, scoring, pos)
ecr_path = fp_ecr_path(year, scoring)

df_stats_raw = read_csv(stats_path)
df_adp_raw = read_csv(adp_path)
df_ecr_raw = read_csv(ecr_path)

display(df_stats_raw.head(3))
display(df_adp_raw.head(3))
display(df_ecr_raw.head(3))

# %% [markdown]
# Clean & standardize

# %%
from utils.cleaners import clean_stats, clean_adp, clean_ecr

df_stats = clean_stats(df_stats_raw)
df_adp = clean_adp(df_adp_raw)
df_ecr = clean_ecr(df_ecr_raw)

display(df_stats.head(3))
display(df_adp.head(3))
display(df_ecr.head(3))

# %% [markdown]
# Merge datasets and compute errors

# %%
from utils.merge import merge_all

df_all = merge_all(df_adp, df_ecr, df_stats)
show(df_all)

# %% [markdown]
# Metrics

# %%
from utils.metrics import compute_summary, top_outperformers, biggest_busts

summary = compute_summary(df_all)
print("Summary:")
for k, v in summary.items():
    print(f"  {k}: {v:.3f}")

print("\nTop outperformers (by ADP vs final):")
show(top_outperformers(df_all, 20))

print("\nBiggest busts:")
show(biggest_busts(df_all, 20))

# %% [markdown]
# Plot

# %%
from utils.plotting import scatter_adp_vs_final

title = f"ADP vs Final Rank — {year.upper()} {scoring.upper()} {pos.upper()}"
scatter_adp_vs_final(df_all, title=title)

# %% [markdown]
# Notes
# - Adjust year/scoring/pos above to analyze other segments.
# - Files must exist in the fp_* directories per naming patterns.
# - This file is a Jupytext percent script; it can be opened as a notebook in VS Code or synced with an .ipynb using jupytext.
