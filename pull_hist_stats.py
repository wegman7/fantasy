# fp_fantasypros_stats_scraper.py
# Scrapes FantasyPros season stats tables (QB/RB/WR/TE/K/DST) for multiple years
# and scoring formats (PPR & Half-PPR). Saves one CSV per (year, scoring, position).
#
# Usage:
#   pip install requests pandas
#   # optional (to reuse browser cookies if needed): pip install browser-cookie3
#   python fp_fantasypros_stats_scraper.py
#
# Output:
#   ./fp_season_stats/fp_stats_{YEAR}_{SCORING}_{POS}.csv
#   e.g., fp_stats_2016_ppr_qb.csv
#
# Notes:
# - Handles MultiIndex headers (e.g., PASSING/RUSHING/MISC row) by flattening.
# - Retries + polite sleep between requests.
# - Public stats pages generally work without login; optional cookie reuse included.

import os
import io
import re
import time
import random
from typing import List, Tuple
from dataclasses import dataclass

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
try:
    # Retry from urllib3 (used by requests)
    from urllib3.util.retry import Retry
except Exception:
    Retry = None  # If missing, we'll still proceed without retry

BASE = "https://www.fantasypros.com"
POSITIONS = ["qb","rb","wr","te","k","dst"]   # what FP uses in the stats URLs
SCORINGS = ["PPR","HALF"]                     # FP query expects "PPR" or "HALF" (standard is "STD")
YEARS = list(range(2002, 2025))               # 2015–2024

DEFAULT_SLEEP = (0.8, 1.8)  # polite rate-limiting window (seconds)


@dataclass
class ScrapeResult:
    df: pd.DataFrame
    url: str


def build_url(pos: str, scoring: str, year: int) -> str:
    """
    Example:
      https://www.fantasypros.com/nfl/stats/qb.php?scoring=PPR&year=2016
      https://www.fantasypros.com/nfl/stats/dst.php?scoring=PPR&year=2016
    """
    return f"{BASE}/nfl/stats/{pos}.php?scoring={scoring}&year={year}"


def session_with_retry(use_browser_cookies: bool = False) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (research script; contact: you@example.com)"
    })
    if Retry is not None:
        retry = Retry(
            total=5,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
    else:
        s.mount("https://", HTTPAdapter())

    return s


# -------- HTML parsing helpers --------

def read_tables(html_text: str):
    # Read from a string (no FutureWarning) and prefer lxml parser for stability
    return pd.read_html(io.StringIO(html_text), displayed_only=False, flavor="lxml")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join([str(x).strip() for x in tup if x and str(x).strip() != ""]).strip()
            for tup in out.columns.to_list()
        ]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out


def pick_fpts_table(tables):
    tokens = ("FPTS", "FPTS/G", "FANTASY POINTS", "FANTASY POINTS PER GAME")
    best = None
    best_cols = -1
    for df in tables:
        df2 = flatten_columns(df)
        # normalize cols: uppercase, remove non-alphanum except '/'
        norm = [re.sub(r"[^A-Z0-9/]", "", c.upper()) for c in df2.columns]
        # accept if ANY token appears as a substring in ANY column
        if any(any(tok.replace(" ", "") in c for c in norm) for tok in tokens):
            if df2.shape[1] > best_cols:  # prefer the widest plausible table
                best = df2
                best_cols = df2.shape[1]
    return best if best is not None else pd.DataFrame()


def normalize_stats_df(
    df: pd.DataFrame, pos_slug: str, year: int, scoring: str, source_url: str
) -> pd.DataFrame:
    out = df.copy()

    # Try to identify player/team/pos/games columns even after flattening
    name_col = next((c for c in out.columns if re.search(r"\bplayer\b|\bname\b", c, re.I)), None)
    team_col = next((c for c in out.columns if re.search(r"\bteam\b", c, re.I)), None)
    pos_col  = next((c for c in out.columns if re.search(r"\bpos(ition)?\b", c, re.I)), None)
    g_col    = next((c for c in out.columns if re.fullmatch(r"g|games", c, flags=re.I)), None)

    if name_col and name_col != "player_name":
        out.rename(columns={name_col: "player_name"}, inplace=True)
    if team_col and team_col != "team":
        out.rename(columns={team_col: "team"}, inplace=True)
    if pos_col and pos_col != "pos":
        out.rename(columns={pos_col: "pos"}, inplace=True)
    if g_col and g_col != "games":
        out.rename(columns={g_col: "games"}, inplace=True)

    # Map fantasy points columns (total and per-game), regardless of exact label
    fpts_col = next((c for c in out.columns if re.search(r"\bfpts\b|\bfantasy\s*points\b", c, re.I)), None)
    fptsg_col = next((c for c in out.columns if re.search(r"\bfpts/?g\b|\bfantasy\s*points.*game\b", c, re.I)), None)

    if fpts_col and fpts_col != "fantasy_points":
        out.rename(columns={fpts_col: "fantasy_points"}, inplace=True)
    if fptsg_col and fptsg_col != "fantasy_points_per_game":
        out.rename(columns={fptsg_col: "fantasy_points_per_game"}, inplace=True)

    # Coerce numerics where sensible
    for c in ("fantasy_points", "fantasy_points_per_game", "games"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Add context columns
    out["season"] = year
    out["scoring"] = scoring.lower()  # "ppr"/"half"
    out["pos_slug"] = pos_slug.upper()
    out["source_url"] = source_url

    # Keep core columns first for convenience
    core = [c for c in [
        "player_name", "team", "pos", "games", "fantasy_points", "fantasy_points_per_game",
        "season", "scoring", "pos_slug", "source_url"
    ] if c in out.columns]
    other = [c for c in out.columns if c not in core]
    return out[core + other]


# -------- Scrape core --------

def scrape_stats(pos: str, scoring: str, year: int, sess: requests.Session):
    url = build_url(pos, scoring, year)
    resp = sess.get(url, timeout=30, allow_redirects=True)

    # treat a real redirect to login as a wall; ignore "Sign In" text in nav
    if "account/login" in resp.url:
        return ScrapeResult(pd.DataFrame(), url)

    resp.raise_for_status()
    tables = read_tables(resp.text)
    table = pick_fpts_table(tables)
    if table.empty:
        return ScrapeResult(pd.DataFrame(), url)
    table = normalize_stats_df(table, pos, year, scoring, url)
    return ScrapeResult(table, url)


def harvest_stats(
    years: List[int] = YEARS,
    scorings: List[str] = SCORINGS,
    positions: List[str] = POSITIONS,
    out_dir: str = "fp_season_stats",
    use_browser_cookies: bool = False,
    sleep_range: Tuple[float, float] = DEFAULT_SLEEP,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sess = session_with_retry(use_browser_cookies=use_browser_cookies)

    for year in years:
        for scoring in scorings:
            for pos in positions:
                try:
                    time.sleep(random.uniform(*sleep_range))
                    result = scrape_stats(pos, scoring, year, sess)
                    if result.df.empty:
                        print(f"[MISS] {year} {scoring} {pos} -> no FPTS table or login wall ({result.url})")
                        continue
                    fname = f"fp_stats_{year}_{scoring.lower()}_{pos}.csv"
                    fpath = os.path.join(out_dir, fname)
                    result.df.to_csv(fpath, index=False)
                    print(f"[OK] {year} {scoring} {pos} -> {fname}")
                except Exception as e:
                    print(f"[ERR] {year} {scoring} {pos}: {e}")


if __name__ == "__main__":
    # Default run: public pages only (no cookies), 2015..2024, PPR & HALF, all positions
    harvest_stats()
    # html = session_with_retry().get("https://www.fantasypros.com/nfl/stats/qb.php?scoring=PPR&year=2016").text
    # tabs = read_tables(html)
    # for i, t in enumerate(tabs):
    #     t2 = flatten_columns(t)
    #     print(i, t2.shape, list(t2.columns)[:12])  # peek at first 12 cols

