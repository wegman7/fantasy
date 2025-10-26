from typing import Dict, Tuple
import pandas as pd


def mae(series: pd.Series) -> float:
    return series.abs().mean()


def compute_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Compute MAE, correlation, and mean error (bias) for ADP and ECR.
    Expects columns: espn_adp, ecr_rank, final_rank, adp_error, ecr_error
    """
    out = {}
    out["mae_adp"] = mae(df["adp_error"])
    out["mae_ecr"] = mae(df["ecr_error"])
    out["corr_adp_final"] = df["espn_adp"].corr(df["final_rank"]) if len(df) else float("nan")
    out["corr_ecr_final"] = df["ecr_rank"].corr(df["final_rank"]) if len(df) else float("nan")
    out["bias_adp"] = df["adp_error"].mean()
    out["bias_ecr"] = df["ecr_error"].mean()
    return out


def top_outperformers(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Players drafted late but finished high: largest positive (final_rank better than ADP) -> most negative adp_error.
    Smaller adp_error (negative) means outperformed ADP.
    """
    return df.sort_values("adp_error").head(n)


def biggest_busts(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Players drafted high but finished low: largest positive adp_error."""
    return df.sort_values("adp_error", ascending=False).head(n)
