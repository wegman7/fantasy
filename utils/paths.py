from pathlib import Path
from typing import Literal


ROOT = Path(__file__).resolve().parents[1]


def fp_stats_path(year: str, scoring: Literal["ppr", "half", "std"], pos: str) -> Path:
    """Build path to season stats CSV.

    Expected pattern: fp_season_stats/fp_stats_{year}_{scoring}_{pos}.csv
    """
    return ROOT / "fp_season_stats" / f"fp_stats_{year}_{scoring}_{pos}.csv"


def fp_adp_path(year: str, scoring: Literal["ppr", "half", "std"], pos: str) -> Path:
    """Build path to ADP CSV.

    Expected pattern: fp_adp/fp_adp_{year}_{scoring}_{pos}.csv
    """
    return ROOT / "fp_adp" / f"fp_adp_{year}_{scoring}_{pos}.csv"


def fp_ecr_path(year: str, scoring: Literal["ppr", "half", "std"]) -> Path:
    """Build path to ECR CSV.

    Example seen in notebook: fp_ecr/FantasyPros_{year}_Draft_ALL_Rankings_{scoring}.csv
    """
    return ROOT / "fp_ecr" / f"FantasyPros_{year}_Draft_ALL_Rankings_{scoring}.csv"
