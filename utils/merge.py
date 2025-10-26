import pandas as pd


def merge_all(df_adp: pd.DataFrame, df_ecr: pd.DataFrame, df_stats: pd.DataFrame) -> pd.DataFrame:
    """Merge ADP, ECR, and final stats by player_name.

    Returns unified dataframe with espn_adp, ecr_rank, final_rank, and error columns.
    """
    df = (
        df_adp
        .merge(df_ecr, on="player_name", how="inner")
        .merge(df_stats, on="player_name", how="inner")
    )
    # Compute initial errors
    df["adp_error"] = df["espn_adp"] - df["final_rank"]
    df["ecr_error"] = df["ecr_rank"] - df["final_rank"]

    # If final ranks are duplicated (common when unioning per-pos ranks),
    # re-rank globally with deterministic tie-breakers so ranks are unique.
    if df["final_rank"].duplicated().any():
        sort_cols = [c for c in ["final_rank", "ecr_rank", "espn_adp", "player_name"] if c in df.columns]
        ascending = [True] * len(sort_cols)
        df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
        df["final_rank"] = pd.RangeIndex(1, len(df) + 1)
        # Recompute errors after re-ranking
        df["adp_error"] = df["espn_adp"] - df["final_rank"]
        df["ecr_error"] = df["ecr_rank"] - df["final_rank"]

    return df
