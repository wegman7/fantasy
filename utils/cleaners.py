import pandas as pd
from typing import Optional


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


def infer_points_column(df: pd.DataFrame) -> Optional[str]:
    """Try to find a column that represents total fantasy points for the season.
    Returns the column name or None if not found.
    """
    candidates_exact = [
        "FPTS", "Fantasy Points", "FantasyPoints", "Total Fantasy Points",
        "PPR Fantasy Points", "PPR Points", "Points", "PTS", "Pts"
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates_exact:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # heuristic: any column containing 'fantasy' and 'point'
    for c in df.columns:
        lc = c.lower()
        if "fantasy" in lc and "point" in lc:
            return c
    # heuristic: 'ppr' sometimes is the points column for PPR exports
    for c in df.columns:
        if c.strip().lower() == "ppr":
            return c
    return None


def clean_stats_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and compute overall (cross-position) final ranks from a union of stats frames.

    - Cleans player names similarly to clean_stats.
    - If a fantasy points column is present, ranks by descending points.
    - Otherwise, falls back to ascending existing rank column if present.
    """
    df = df.copy()
    # player name
    player_col: Optional[str] = None
    if "Unnamed: 1_level_0_Player" in df.columns:
        df["Unnamed: 1_level_0_Player"] = df["Unnamed: 1_level_0_Player"].astype(str).str.replace(r"\s*\(.*\)", "", regex=True)
        player_col = "Unnamed: 1_level_0_Player"
    else:
        for c in ["Player", "PLAYER", "player", "player_name"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(r"\s*\(.*\)", "", regex=True)
                player_col = c
                break
    if not player_col:
        raise KeyError("Could not find player name column in stats dataframe for overall computation")

    # try fantasy points first
    pts_col = infer_points_column(df)
    if pts_col and pts_col in df.columns:
        pts = pd.to_numeric(df[pts_col], errors="coerce").fillna(0)
        # Rank descending by points; method='first' ensures deterministic order
        ranks = pts.rank(ascending=False, method="first").astype(int)
        out = df[[player_col]].copy()
        out["final_rank"] = ranks.values
        out = out.sort_values("final_rank", kind="mergesort")
        out = out.rename(columns={player_col: "player_name"})
        return out

    # fallback to existing rank column if available
    if "Unnamed: 0_level_0_Rank" in df.columns:
        rank_series = pd.to_numeric(df["Unnamed: 0_level_0_Rank"], errors="coerce")
        out = df[[player_col]].copy()
        out["final_rank"] = rank_series
        out = out.sort_values("final_rank", kind="mergesort")
        out = out.rename(columns={player_col: "player_name"})
        return out

    # last resort: alphabetical rank
    out = df[[player_col]].copy()
    out = out.rename(columns={player_col: "player_name"})
    out = out.sort_values("player_name", kind="mergesort")
    out["final_rank"] = pd.RangeIndex(1, len(out) + 1)
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
        for c in ["ADP", "adp", "espn_adp", "overall_adp", "ovr_adp", "adp_overall"]:
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
