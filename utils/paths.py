from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fp_stats_path(year: str, scoring: str, pos: str) -> Path:
    """Build path to season stats CSV.

    Expected pattern: fp_season_stats/fp_stats_{year}_{scoring}_{pos}.csv
    """
    return ROOT / "fp_season_stats" / f"fp_stats_{year}_{scoring}_{pos}.csv"


def fp_adp_path(year: str, scoring: str, pos: str) -> Path:
    """Build path to ADP CSV.

    Expected pattern: fp_adp/fp_adp_{year}_{scoring}_{pos}.csv
    """
    return ROOT / "fp_adp" / f"fp_adp_{year}_{scoring}_{pos}.csv"


def fp_ecr_path(year: str, scoring: str) -> Path:
    """Build path to ECR CSV.

    Example seen in notebook: fp_ecr/FantasyPros_{year}_Draft_ALL_Rankings_{scoring}.csv
    """
    return ROOT / "fp_ecr" / f"FantasyPros_{year}_Draft_ALL_Rankings_{scoring}.csv"


def fp_adp_overall_path(year: str, scoring: str) -> Path:
    """Build path to overall ADP CSV (ovr).

    Pattern: fp_adp/fp_adp_{year}_{scoring}_overall.csv
    """
    return ROOT / "fp_adp" / f"fp_adp_{year}_{scoring}_overall.csv"


def available_years(scoring: str, pos: str) -> list[str]:
    """Return sorted list of years that have stats files for the given scoring and pos."""
    pattern = f"fp_stats_*_{scoring}_{pos}.csv"
    years = []
    for p in (ROOT / "fp_season_stats").glob(pattern):
        # filename: fp_stats_{year}_{scoring}_{pos}.csv
        name = p.stem
        parts = name.split("_")
        if len(parts) >= 5 and parts[0] == "fp" and parts[1] == "stats":
            years.append(parts[2])
    years.remove('2022')
    years.remove('2023')
    return sorted(set(years))
