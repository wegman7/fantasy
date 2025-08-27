# fp_adp_ecr_scraper.py
# Grab FantasyPros historical ADP (with per-source columns like ESPN/Yahoo)
# and historical ECR (Expert Consensus Rankings) for multiple years/scorings.
#
# Usage:
#   pip install requests pandas
#   python fp_adp_ecr_scraper.py
#
# Output:
#   ./fp_adp/fp_adp_{year}_{scoring}_{pos}.csv
#   ./fp_ecr/fp_ecr_{year}_{scoring}_{pos}.csv
#   ./fp_join/fp_adp_ecr_{year}_{scoring}_{pos}.csv          (optional join)
#
# Notes:
# - ADP pages expose per-source columns (e.g., ESPN on PPR pages; Yahoo on Half-PPR).
# - ECR “cheatsheets” pages are year-addressable.
# - We read the HTML tables directly (no need to press the site’s “Download CSV”).
# - Handles multi-row headers via flattening; picks the widest plausible table.

import os
import io
import re
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except Exception:
    Retry = None
from urllib.parse import urlencode

BASE = "https://www.fantasypros.com"
YEARS = list(range(2015, 2026))          # 2015..2024
SCORINGS = ["PPR", "HALF"]               # Standard available too if you add it
POSITIONS = ["overall", "qb", "rb", "wr", "te", "k", "dst"]

DEFAULT_SLEEP = (0.7, 1.4)

# -------------- Session / helpers --------------

def session_with_retry() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (research script)"})
    if Retry is not None:
        r = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
        )
        s.mount("https://", HTTPAdapter(max_retries=r))
    else:
        s.mount("https://", HTTPAdapter())
    return s

def read_tables(html_text: str) -> List[pd.DataFrame]:
    # Avoid FutureWarning by wrapping literal HTML
    return pd.read_html(io.StringIO(html_text), displayed_only=False)

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

# -------------- ADP --------------

def adp_slug(scoring: str, pos: str) -> str:
    # PPR:  ppr-overall.php, ppr-wr.php, ...
    # HALF: half-point-ppr-overall.php, half-point-ppr-wr.php, ...
    base = "ppr" if scoring.upper() == "PPR" else "half-point-ppr"
    return f"{base}-{pos}"

def build_adp_url(scoring: str, pos: str, year: int) -> str:
    slug = adp_slug(scoring, pos)
    return f"{BASE}/nfl/adp/{slug}.php?year={year}"

def pick_adp_table(tables: List[pd.DataFrame]) -> pd.DataFrame:
    # Accept a table that has typical ADP columns:
    # per-source names (ESPN, YAHOO, SLEEPER, CBS, NFL, RTSports, Fantrax) and/or AVG
    tokens = ("ESPN", "YAHOO", "SLEEPER", "CBS", "NFL", "RTSPORTS", "FANTRAX", "AVG", "ADP")
    best = None
    best_cols = -1
    for df in tables:
        df2 = flatten_columns(df)
        norm = [re.sub(r"[^A-Z0-9/]", "", c.upper()) for c in df2.columns]
        if any(any(tok in c for tok in tokens) for c in norm):
            if df2.shape[1] > best_cols:
                best, best_cols = df2, df2.shape[1]
    return best if best is not None else pd.DataFrame()

def normalize_adp_df(df: pd.DataFrame, pos: str, scoring: str, year: int, url: str) -> pd.DataFrame:
    out = df.copy()

    # Player + team/pos name columns
    name_col = next((c for c in out.columns if re.search(r"\bplayer\b|\bname\b", c, re.I)), None)
    team_col = next((c for c in out.columns if re.fullmatch(r"team(\s*\(bye\))?", c, flags=re.I)), None)
    pos_col  = next((c for c in out.columns if re.fullmatch(r"pos(ition)?", c, flags=re.I)), None)

    if name_col and name_col != "player_name": out.rename(columns={name_col: "player_name"}, inplace=True)
    if team_col and team_col != "team":        out.rename(columns={team_col: "team"}, inplace=True)
    if pos_col and pos_col != "pos":           out.rename(columns={pos_col: "pos"}, inplace=True)

    # Common numeric columns to standardize
    ren_map = {}
    for c in list(out.columns):
        cu = c.upper()
        if cu == "AVG":
            ren_map[c] = "adp_avg"
        elif cu == "ESPN":
            ren_map[c] = "adp_espn"
        elif cu == "YAHOO":
            ren_map[c] = "adp_yahoo"
        elif cu == "SLEEPER":
            ren_map[c] = "adp_sleeper"
        elif cu == "CBS":
            ren_map[c] = "adp_cbs"
        elif cu == "NFL":
            ren_map[c] = "adp_nfl"
        elif cu in ("RTSPORTS", "RTSPORTS"):  # just in case
            ren_map[c] = "adp_rtsports"
        elif "FANTRAX" in cu:
            ren_map[c] = "adp_fantrax"
        elif cu in ("ADP", "AVGADP"):
            ren_map[c] = "adp_avg"

    out.rename(columns=ren_map, inplace=True)

    # Coerce numeric on any ADP columns we recognized
    for c in [col for col in out.columns if col.startswith("adp_")]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["season"] = year
    out["scoring"] = scoring.lower()
    out["pos_slug"] = pos.upper()
    out["source_url"] = url

    core = [c for c in ["player_name", "team", "pos", "adp_espn", "adp_yahoo", "adp_avg",
                        "season", "scoring", "pos_slug", "source_url"] if c in out.columns]
    other = [c for c in out.columns if c not in core]
    return out[core + other]

def harvest_adp(years=YEARS, scorings=SCORINGS, positions=POSITIONS, out_dir="fp_adp"):
    os.makedirs(out_dir, exist_ok=True)
    sess = session_with_retry()
    for y in years:
        for s in scorings:
            for p in positions:
                try:
                    url = build_adp_url(s, p, y)
                    r = sess.get(url, timeout=30)
                    if "account/login" in r.url:
                        print(f"[MISS] ADP {y} {s} {p}: login wall ({url})")
                        continue
                    tbls = read_tables(r.text)
                    table = pick_adp_table(tbls)
                    if table is None or table.empty:
                        print(f"[MISS] ADP {y} {s} {p}: no ADP table ({url})")
                        continue
                    df = normalize_adp_df(table, p, s, y, url)
                    fname = f"fp_adp_{y}_{s.lower()}_{p}.csv"
                    df.to_csv(os.path.join(out_dir, fname), index=False)
                    print(f"[OK]   ADP {y} {s} {p} -> {fname}")
                    time.sleep(random.uniform(*DEFAULT_SLEEP))
                except Exception as e:
                    print(f"[ERR]  ADP {y} {s} {p}: {e}")

# -------------- ECR --------------

from urllib.parse import urlencode

# Map our position names to FantasyPros query values
_POS_MAP = {
    "overall": "ALL", "qb": "QB", "rb": "RB", "wr": "WR",
    "te": "TE", "k": "K", "dst": "DST", "flex": "FLEX"
}

def _build_ecr_attempt_urls(year: int, scoring: str, pos: str) -> list[str]:
    scoring = scoring.upper()
    pos_l = pos.lower()

    # A) Newer, stable rankings index with query params
    qsA = {"type": "draft", "scoring": scoring, "year": year, "position": _POS_MAP.get(pos_l, "ALL")}
    urlA = f"{BASE}/nfl/rankings/?{urlencode(qsA)}"

    # B) consensus-cheatsheets with scoring/year (and position when not overall)
    qsB = {"year": year, "scoring": scoring}
    if pos_l != "overall":
        qsB["position"] = _POS_MAP.get(pos_l, "ALL")
    urlB = f"{BASE}/nfl/rankings/consensus-cheatsheets.php?{urlencode(qsB)}"

    # C) Legacy cheatsheets paths (ppr-/half-point-ppr- prefixes)
    if scoring == "PPR":
        prefix = "ppr-"
    elif scoring in ("HALF", "HALF_PPR", "HALF-POINT-PPR"):
        prefix = "half-point-ppr-"
    else:
        prefix = ""
    if pos_l == "overall":
        path = f"/nfl/rankings/{prefix}cheatsheets.php"
    else:
        path = f"/nfl/rankings/{prefix}{pos_l}-cheatsheets.php"
    urlC = f"{BASE}{path}?year={year}"

    return [urlA, urlB, urlC]

def _pick_ecr_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Pick the widest table that looks like rankings (has Rank + Player)."""
    best, best_cols = None, -1
    for df in tables:
        df2 = flatten_columns(df)
        cols_up = [str(c).upper() for c in df2.columns]
        has_player = any(("PLAYER" in c) or ("NAME" in c) for c in cols_up)
        has_rank   = any((c in ("RK", "RANK")) or ("RANK" in c) for c in cols_up)
        if has_player and has_rank:
            if df2.shape[1] > best_cols:
                best, best_cols = df2, df2.shape[1]
    return best if best is not None else pd.DataFrame()

def fetch_ecr(year: int, scoring: str, pos: str, sess: requests.Session) -> tuple[pd.DataFrame, str]:
    """Try multiple URL patterns; return (table, url_used) or (empty, '')."""
    for url in _build_ecr_attempt_urls(year, scoring, pos):
        resp = sess.get(url, timeout=25, allow_redirects=True)
        if "account/login" in resp.url:  # real login redirect
            continue
        try:
            tables = read_tables(resp.text)
        except ValueError:
            continue
        table = _pick_ecr_table(tables)
        if not table.empty:
            return table, url
    return pd.DataFrame(), ""

def normalize_ecr_df(df: pd.DataFrame, pos: str, scoring: str, year: int, url: str) -> pd.DataFrame:
    out = df.copy()
    # Rename common columns if present
    name_col = next((c for c in out.columns if re.search(r"\bplayer\b|\bname\b", c, re.I)), None)
    team_col = next((c for c in out.columns if re.fullmatch(r"team(\s*\(bye\))?", c, flags=re.I)), None)
    pos_col  = next((c for c in out.columns if re.fullmatch(r"pos(ition)?", c, flags=re.I)), None)
    rk_col   = next((c for c in out.columns if re.fullmatch(r"rk|rank", c, flags=re.I)), None)
    ecr_col  = next((c for c in out.columns if re.fullmatch(r"ecr", c, flags=re.I)), None)

    if name_col and name_col != "player_name": out.rename(columns={name_col: "player_name"}, inplace=True)
    if team_col and team_col != "team":        out.rename(columns={team_col: "team"}, inplace=True)
    if pos_col and pos_col != "pos":           out.rename(columns={pos_col: "pos"}, inplace=True)
    if rk_col and rk_col != "ecr_rank":        out.rename(columns={rk_col: "ecr_rank"}, inplace=True)
    if ecr_col and ecr_col != "ecr":           out.rename(columns={ecr_col: "ecr"}, inplace=True)

    # Coerce numerics
    for c in ("ecr_rank", "ecr"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["season"]  = year
    out["scoring"] = scoring.lower()
    out["pos_slug"] = pos.upper()
    out["source_url"] = url

    core = [c for c in ["player_name","team","pos","ecr_rank","ecr","season","scoring","pos_slug","source_url"] if c in out.columns]
    other = [c for c in out.columns if c not in core]
    return out[core + other]

def harvest_ecr(
    years=YEARS, scorings=SCORINGS, positions=["overall","qb","rb","wr","te","k","dst"], out_dir="fp_ecr"
):
    os.makedirs(out_dir, exist_ok=True)
    sess = session_with_retry()
    for y in years:
        for s in scorings:
            for p in positions:
                try:
                    time.sleep(random.uniform(*DEFAULT_SLEEP))
                    table, used = fetch_ecr(y, s, p, sess)
                    if table.empty:
                        print(f"[MISS] ECR {y} {s} {p}: no table ({used or 'no-parse'})")
                        continue
                    df = normalize_ecr_df(table, p, s, y, used)
                    fname = f"fp_ecr_{y}_{s.lower()}_{p}.csv"
                    df.to_csv(os.path.join(out_dir, fname), index=False)
                    print(f"[OK]   ECR {y} {s} {p} -> {fname}")
                except Exception as e:
                    print(f"[ERR]  ECR {y} {s} {p}: {e}")


# -------------- Optional: join ADP & ECR --------------

def join_adp_ecr(year: int, scoring: str, pos: str,
                 adp_dir="fp_adp", ecr_dir="fp_ecr", out_dir="fp_join") -> str:
    os.makedirs(out_dir, exist_ok=True)
    adp_path = os.path.join(adp_dir, f"fp_adp_{year}_{scoring.lower()}_{pos}.csv")
    ecr_path = os.path.join(ecr_dir, f"fp_ecr_{year}_{scoring.lower()}_{pos}.csv")
    if not (os.path.exists(adp_path) and os.path.exists(ecr_path)):
        return ""

    adp = pd.read_csv(adp_path)
    ecr = pd.read_csv(ecr_path)

    # Light cleanup: normalize player names to aid matching
    def clean_name(s):
        if pd.isna(s):
            return s
        s = re.sub(r"\s+\(.*?\)$", "", str(s))  # strip trailing (Team) variants if any
        return re.sub(r"\s+", " ", s).strip()

    for df in (adp, ecr):
        if "player_name" in df.columns:
            df["player_key"] = df["player_name"].map(clean_name).str.lower()
        else:
            df["player_key"] = pd.NA

    on_cols = ["player_key"]
    merged = pd.merge(ecr, adp, on=on_cols, how="inner", suffixes=("_ecr", "_adp"))

    out_path = os.path.join(out_dir, f"fp_adp_ecr_{year}_{scoring.lower()}_{pos}.csv")
    merged.to_csv(out_path, index=False)
    return out_path

# -------------- Run all --------------

if __name__ == "__main__":
    # harvest_adp()
    harvest_ecr()

    # Example: build a few joined files (overall + WR)
    # for y in YEARS:
    #     for s in SCORINGS:
    #         for p in ["overall", "wr"]:
    #             path = join_adp_ecr(y, s, p)
    #             if path:
    #                 print(f"[OK] JOIN {y} {s} {p} -> {path}")
