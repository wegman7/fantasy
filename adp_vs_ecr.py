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
pos = "all"      # "all" or one of [qb, rb, wr, te] or a list like ["wr","rb"]

# %% [markdown]
# Load data

# %%
from utils.paths import fp_stats_path, fp_adp_path, fp_ecr_path, fp_adp_overall_path
from utils.loaders import read_csv

def normalize_positions(p):
    if isinstance(p, str):
        if p.lower() == "all":
            return ["wr", "rb", "qb", "te"]
        return [p.lower()]
    return [x.lower() for x in p]

positions = normalize_positions(pos)

ecr_path = fp_ecr_path(year, scoring)
df_ecr_raw = read_csv(ecr_path)

if len(positions) == 1:
    stats_path = fp_stats_path(year, scoring, positions[0])
    adp_path = fp_adp_path(year, scoring, positions[0])
    df_stats_raw = read_csv(stats_path)
    df_adp_raw = read_csv(adp_path)
else:
    stats_list = []
    for p in positions:
        sp = fp_stats_path(year, scoring, p)
        stats_list.append(read_csv(sp).assign(pos=p))
    df_stats_raw = pd.concat(stats_list, ignore_index=True)
    # Use overall ADP (ovr) when analyzing multiple positions
    df_adp_raw = read_csv(fp_adp_overall_path(year, scoring))

display(df_stats_raw.head(3))
display(df_adp_raw.head(3))
display(df_ecr_raw.head(3))

# %% [markdown]
# Clean & standardize

# %%
from utils.cleaners import clean_stats, clean_adp, clean_ecr, clean_stats_overall

if len(positions) == 1:
    df_stats = clean_stats(df_stats_raw)
    df_adp = clean_adp(df_adp_raw)
else:
    # Compute overall final ranks across all positions
    df_stats = clean_stats_overall(df_stats_raw)
    # ADP overall already loaded; clean
    df_adp = clean_adp(df_adp_raw)

df_stats = df_stats.drop_duplicates(subset=["player_name"])  # safety
df_adp = df_adp.drop_duplicates(subset=["player_name"])      # safety
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
from utils.plotting import scatter_adp_vs_final, scatter_ecr_vs_final

pos_label = ("ALL" if len(positions) > 1 else positions[0].upper())
title_adp = f"ADP vs Final Rank — {year.upper()} {scoring.upper()} {pos_label}"
scatter_adp_vs_final(df_all, title=title_adp)

title_ecr = f"ECR vs Final Rank — {year.upper()} {scoring.upper()} {pos_label}"
scatter_ecr_vs_final(df_all, title=title_ecr)

# %% [markdown]
# Notes
# - Adjust year/scoring/pos above to analyze other segments.
# - Files must exist in the fp_* directories per naming patterns.
# - This file is a Jupytext percent script; it can be opened as a notebook in VS Code or synced with an .ipynb using jupytext.
