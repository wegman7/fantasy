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

    # Compute errors
    df["adp_error"] = df["espn_adp"] - df["final_rank"]
    df["ecr_error"] = df["ecr_rank"] - df["final_rank"]
    return df
