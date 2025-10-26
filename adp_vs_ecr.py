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
year = "all"  # "all", a single year like "2024", or a list of years
scoring = "ppr"  # ppr | half | std
pos = "all"      # "all" or one of [qb, rb, wr, te] or a list like ["wr","rb"]

# %% [markdown]
# Load data

# %%
from utils.paths import fp_stats_path, fp_adp_path, fp_ecr_path, fp_adp_overall_path, available_years
from utils.loaders import read_csv

def normalize_positions(p):
    if isinstance(p, str):
        if p.lower() == "all":
            return ["wr", "rb", "qb", "te"]
        return [p.lower()]
    return [x.lower() for x in p]

positions = normalize_positions(pos)

# Resolve years to load
def normalize_years(y):
    if isinstance(y, str):
        if y.lower() == "all":
            # Use years available for WR (as proxy) and filter to 2015+
            yrs = available_years(scoring, "wr")
            return sorted([yy for yy in yrs if int(yy) >= 2015])
        return [y]
    return [str(x) for x in y]

years = normalize_years(year)

stats_frames = []
adp_frames = []
ecr_frames = []

for y in years:
    if len(positions) == 1:
        stats_frames.append(read_csv(fp_stats_path(y, scoring, positions[0])).assign(year=y))
        adp_frames.append(read_csv(fp_adp_path(y, scoring, positions[0])).assign(year=y))
    else:
        # union stats across positions for each year
        per_year_stats = []
        for p in positions:
            per_year_stats.append(read_csv(fp_stats_path(y, scoring, p)).assign(pos=p, year=y))
        stats_frames.append(pd.concat(per_year_stats, ignore_index=True))
        # overall ADP for that year
        adp_frames.append(read_csv(fp_adp_overall_path(y, scoring)).assign(year=y))
    # ECR for that year
    ecr_frames.append(read_csv(fp_ecr_path(y, scoring)).assign(year=y))

df_stats_raw = pd.concat(stats_frames, ignore_index=True)
df_adp_raw = pd.concat(adp_frames, ignore_index=True)
df_ecr_raw = pd.concat(ecr_frames, ignore_index=True)

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
    # Compute overall final ranks across all positions, per year if multi-year
    if len(years) > 1:
        df_stats = (
            df_stats_raw.groupby("year", group_keys=False).apply(clean_stats_overall)
        )
        df_stats["year"] = df_stats["year"].astype(str)
    else:
        df_stats = clean_stats_overall(df_stats_raw)
        df_stats["year"] = years[0]
    # ADP overall already loaded; clean
    df_adp = clean_adp(df_adp_raw)

subset_cols = ["player_name", "year"] if "year" in df_stats.columns else ["player_name"]
df_stats = df_stats.drop_duplicates(subset=subset_cols)
df_adp = df_adp.drop_duplicates(subset=subset_cols)
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
years_label = ("ALL" if len(years) > 1 else years[0])
title_adp = f"ADP vs Final Rank — {years_label} {scoring.upper()} {pos_label}"
scatter_adp_vs_final(df_all, title=title_adp)

title_ecr = f"ECR vs Final Rank — {years_label} {scoring.upper()} {pos_label}"
scatter_ecr_vs_final(df_all, title=title_ecr)

# %% [markdown]
# Notes
# - Adjust year/scoring/pos above to analyze other segments.
# - Files must exist in the fp_* directories per naming patterns.
# - This file is a Jupytext percent script; it can be opened as a notebook in VS Code or synced with an .ipynb using jupytext.
