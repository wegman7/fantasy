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

# %%
from itables import init_notebook_mode, show
import pandas as pd
init_notebook_mode(all_interactive=True)

# %%
year, scoring, pos = '2024', 'ppr', 'wr'
path = f'/Users/challenger/prog/fantasy/fp_season_stats/fp_stats_{year}_{scoring}_{pos}.csv'

df_stats = pd.read_csv(path)

# %%
# Remove parentheses and text inside them
df_stats["Unnamed: 1_level_0_Player"] = (
    df_stats["Unnamed: 1_level_0_Player"]
    .str.replace(r"\s*\(.*\)", "", regex=True)
)
df_stats = df_stats[["Unnamed: 1_level_0_Player", "Unnamed: 0_level_0_Rank"]]

df_stats

# %%
year, scoring, pos = '2024', 'ppr', 'wr'
path = f'/Users/challenger/prog/fantasy/fp_adp/fp_adp_{year}_{scoring}_{pos}.csv'

df_adp = pd.read_csv(path)

# %%
df_adp["player_name"] = (
    df_adp["player_name"]
    .str.replace(r"\s+[A-Z]{2,3}\s*\(\d+\)", "", regex=True)
)
df_adp = df_adp[["player_name", "adp_espn"]]

df_adp

# %%
year, scoring, pos = '2024', 'ppr', 'wr'
path = f'/Users/challenger/prog/fantasy/fp_ecr/FantasyPros_2024_Draft_ALL_Rankings_ppr.csv'

df_ecr = pd.read_csv(path)
df_ecr

# %%
df_ecr = df_ecr[["PLAYER NAME", "RK"]]

df_ecr

# %%
df_adp_and_stats = (
    df_adp
    .merge(
        df_stats, left_on=["player_name"], right_on=["Unnamed: 1_level_0_Player"], suffixes=("_adp", "_stats")
    )
    .merge(
        df_ecr, left_on=["player_name"], right_on=["PLAYER NAME"]
    )
)
df_adp_and_stats = df_adp_and_stats[df_adp_and_stats['adp_espn'].notna()]
df_adp_and_stats

# %%
df = df_adp_and_stats.rename(columns={
    "adp_espn": "espn_adp",
    "Unnamed: 0_level_0_Rank": "final_rank",
    "PLAYER NAME": "player_name_ecr",
    "RK": "ecr_rank"
})
df["adp_error"] = df["espn_adp"] - df["final_rank"]
df["ecr_error"] = df["ecr_rank"] - df["final_rank"]
mae_adp = (df["adp_error"].abs()).mean()
mae_ecr = (df["ecr_error"].abs()).mean()
print("Mean Absolute Error adp:", mae_adp)
print("Mean Absolute Error ecr:", mae_ecr)
corr = df["espn_adp"].corr(df["final_rank"])
print("Correlation:", corr)
mean_error = df["adp_error"].mean()
print("Mean Error (bias):", mean_error)
df



# %%
import matplotlib.pyplot as plt

plt.scatter(df["espn_adp"], df["final_rank"])
plt.plot([0, max(df["espn_adp"])], [0, max(df["espn_adp"])], "r--")  # perfect line
plt.xlabel("ESPN ADP (expected)")
plt.ylabel("Final Rank (actual)")
plt.title("ESPN ADP vs Actual Performance")
plt.show()


# %%
# Top outperformers (drafted late, finished high)
busts = df.sort_values("adp_error")

# Biggest busts (drafted high, finished low)
outperformers = df.sort_values("adp_error", ascending=False)

show(outperformers)
show(busts)

