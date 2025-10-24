import pandas as pd


def clean_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Clean season stats DataFrame to two columns: player_name, final_rank.

    Observed columns in notebook:
    - Player column: "Unnamed: 1_level_0_Player" with possible parentheses info to drop
    - Rank column: "Unnamed: 0_level_0_Rank"
    """
    df = df.copy()
    if "Unnamed: 1_level_0_Player" in df.columns:
        df["Unnamed: 1_level_0_Player"] = df["Unnamed: 1_level_0_Player"].str.replace(r"\s*\(.*\)", "", regex=True)
        player_col = "Unnamed: 1_level_0_Player"
    else:
        # Fallbacks: try common names
        for c in ["Player", "PLAYER", "player", "player_name"]:
            if c in df.columns:
                player_col = c
                break
        else:
            raise KeyError("Could not find player name column in stats dataframe")

    if "Unnamed: 0_level_0_Rank" in df.columns:
        rank_col = "Unnamed: 0_level_0_Rank"
    else:
        for c in ["Rank", "RK", "rank"]:
            if c in df.columns:
                rank_col = c
                break
        else:
            raise KeyError("Could not find rank column in stats dataframe")

    out = df[[player_col, rank_col]].rename(columns={player_col: "player_name", rank_col: "final_rank"})
    return out


def clean_adp(df: pd.DataFrame) -> pd.DataFrame:
    """Clean ADP DataFrame to two columns: player_name, espn_adp.

    Observed columns in notebook:
    - player_name with suffix like " LAR (123)" to strip team/pos/overall
    - adp_espn numeric
    """
    df = df.copy()
    # player name cleanup
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].str.replace(r"\s+[A-Z]{2,3}\s*\(\d+\)", "", regex=True)
        player_col = "player_name"
    else:
        # try common variants
        for c in ["Player", "PLAYER", "name"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(r"\s+[A-Z]{2,3}\s*\(\d+\)", "", regex=True)
                player_col = c
                break
        else:
            raise KeyError("Could not find player name column in ADP dataframe")

    # adp column
    if "adp_espn" in df.columns:
        adp_col = "adp_espn"
    else:
        for c in ["ADP", "adp", "espn_adp"]:
            if c in df.columns:
                adp_col = c
                break
        else:
            raise KeyError("Could not find ADP column in ADP dataframe")

    out = df[[player_col, adp_col]].rename(columns={player_col: "player_name", adp_col: "espn_adp"})
    out = out[out["espn_adp"].notna()]
    return out


def clean_ecr(df: pd.DataFrame) -> pd.DataFrame:
    """Clean ECR DataFrame to two columns: player_name, ecr_rank.

    Observed columns in notebook:
    - PLAYER NAME
    - RK
    """
    df = df.copy()
    if "PLAYER NAME" in df.columns:
        player_col = "PLAYER NAME"
    else:
        for c in ["Player", "PLAYER", "player_name", "Name"]:
            if c in df.columns:
                player_col = c
                break
        else:
            raise KeyError("Could not find player name in ECR dataframe")

    if "RK" in df.columns:
        rank_col = "RK"
    else:
        for c in ["Rank", "ECR", "rank"]:
            if c in df.columns:
                rank_col = c
                break
        else:
            raise KeyError("Could not find rank column in ECR dataframe")

    out = df[[player_col, rank_col]].rename(columns={player_col: "player_name", rank_col: "ecr_rank"})
    return out
