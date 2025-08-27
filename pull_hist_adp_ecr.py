# fp_adp_ecr_scraper.py
# Grab FantasyPros historical ADP (with per-source columns like ESPN/Yahoo)
# and historical ECR (Expert Consensus Rankings) for multiple years/scorings.
#
# IMPORTANT (per your requirement):
#   ECR is fetched ONLY from these two endpoints:
#     - PPR:  https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php?year=YYYY
#     - HALF: https://www.fantasypros.com/nfl/rankings/half-point-ppr-cheatsheets.php?year=YYYY
#   We fetch "overall" once per (year, scoring) and then FILTER to QB/RB/WR/TE/K/DST.
#
# Usage:
#   pip install requests pandas beautifulsoup4 lxml
#   python fp_adp_ecr_scraper.py
#
# Output:
#   ./fp_adp/fp_adp_{year}_{scoring}_{pos}.csv
#   ./fp_ecr/fp_ecr_{year}_{scoring}_{pos}.csv

import os
import io
import re
import json
import time
import random
from typing import List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except Exception:
    Retry = None

# ---------------------- config ----------------------

BASE = "https://www.fantasypros.com"
YEARS = list(range(2015, 2026))      # 2015..2025
SCORINGS = ["PPR", "HALF"]           # add "STD" if you want standard scoring
POSITIONS = ["overall", "qb", "rb", "wr", "te", "k", "dst"]

DEFAULT_SLEEP = (0.7, 1.4)           # polite rate-limiting

# ---------------------- core helpers ----------------------

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

# =========================================================
# ADP (unchanged, uses the standard ADP endpoints)
# =========================================================

def adp_slug(scoring: str, pos: str) -> str:
    # PPR:  ppr-overall.php, ppr-wr.php, ...
    # HALF: half-point-ppr-overall.php, half-point-ppr-wr.php, ...
    base = "ppr" if scoring.upper() == "PPR" else "half-point-ppr"
    return f"{base}-{pos}"

def build_adp_url(scoring: str, pos: str, year: int) -> str:
    slug = adp_slug(scoring, pos)
    return f"{BASE}/nfl/adp/{slug}.php?year={year}"

def pick_adp_table(tables: List[pd.DataFrame]) -> pd.DataFrame:
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
    name_col = next((c for c in out.columns if re.search(r"\bplayer\b|\bname\b", c, re.I)), None)
    team_col = next((c for c in out.columns if re.fullmatch(r"team(\s*\(bye\))?", c, flags=re.I)), None)
    pos_col  = next((c for c in out.columns if re.fullmatch(r"pos(ition)?", c, flags=re.I)), None)

    if name_col and name_col != "player_name": out.rename(columns={name_col: "player_name"}, inplace=True)
    if team_col and team_col != "team":        out.rename(columns={team_col: "team"}, inplace=True)
    if pos_col and pos_col != "pos":           out.rename(columns={pos_col: "pos"}, inplace=True)

    ren_map = {}
    for c in list(out.columns):
        cu = c.upper()
        if cu == "AVG": ren_map[c] = "adp_avg"
        elif cu == "ESPN": ren_map[c] = "adp_espn"
        elif cu == "YAHOO": ren_map[c] = "adp_yahoo"
        elif cu == "SLEEPER": ren_map[c] = "adp_sleeper"
        elif cu == "CBS": ren_map[c] = "adp_cbs"
        elif cu == "NFL": ren_map[c] = "adp_nfl"
        elif "FANTRAX" in cu: ren_map[c] = "adp_fantrax"
        elif cu in ("ADP", "AVGADP"): ren_map[c] = "adp_avg"
        elif cu in ("RTSPORTS",): ren_map[c] = "adp_rtsports"
    out.rename(columns=ren_map, inplace=True)

    for c in [col for col in out.columns if col.startswith("adp_")]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["season"] = year
    out["scoring"] = scoring.lower()
    out["pos_slug"] = pos.upper()
    out["source_url"] = url

    core = [c for c in ["player_name","team","pos","adp_espn","adp_yahoo","adp_avg",
                        "season","scoring","pos_slug","source_url"] if c in out.columns]
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

# =========================================================
# ECR (ONLY the two cheatsheets endpoints you specified)
# =========================================================

def _ecr_url(scoring: str, year: int) -> str:
    if scoring.upper() == "PPR":
        return f"{BASE}/nfl/rankings/ppr-cheatsheets.php?year={year}"
    elif scoring.upper() == "HALF":
        return f"{BASE}/nfl/rankings/half-point-ppr-cheatsheets.php?year={year}"
    else:
        # default to PPR if someone passes unknown; or raise
        return f"{BASE}/nfl/rankings/ppr-cheatsheets.php?year={year}"

# Embedded JSON patterns FP commonly uses on cheatsheets pages
ECR_JSON_PATTERNS = [
    r"ecrData\s*=\s*(\{.*?\});",
    r"cheatsheetsData\s*=\s*(\{.*?\});",
    r"cheatsheetData\s*=\s*(\{.*?\});",
    r"rankingsData\s*=\s*(\{.*?\});",
]

def _try_extract_ecr_json(html: str):
    for pat in ECR_JSON_PATTERNS:
        m = re.search(pat, html, flags=re.DOTALL)
        if not m:
            continue
        raw = m.group(1).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raw2 = re.sub(r"(?<!\\)'", '"', raw)
            try:
                return json.loads(raw2)
            except Exception:
                pass
    return None

def _ecr_json_to_df(payload: dict) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    players = None
    for key in ("players", "rows", "data", "rankings"):
        if key in payload and isinstance(payload[key], list):
            players = payload[key]
            break
    if players is None:
        for v in payload.values():
            if isinstance(v, dict) and "players" in v and isinstance(v["players"], list):
                players = v["players"]
                break
    if players is None:
        return pd.DataFrame()

    df = pd.DataFrame(players)
    # normalize column names
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("player_name", "name", "player", "playername"):
            rename[c] = "player_name"
        elif cl in ("team", "nflteam"):
            rename[c] = "team"
        elif cl in ("pos", "position", "player_position"):
            rename[c] = "pos"
        elif cl in ("rk", "rank", "rank_ecr", "ecr_rank"):
            rename[c] = "ecr_rank"
        elif cl in ("ecr", "avg", "average"):
            rename[c] = "ecr"
    df = df.rename(columns=rename)
    return df

def _read_rank_table_from_dom(html: str) -> pd.DataFrame:
    """Grab ONLY the big rankings table: <table id='ranking-table'> … (ignores 5-row widgets)."""
    soup = BeautifulSoup(html, "lxml")
    tbl = soup.select_one("table#ranking-table")
    if not tbl:
        return pd.DataFrame()
    dfs = pd.read_html(io.StringIO(str(tbl)), displayed_only=False)
    dfs.sort(key=lambda d: (len(d), d.shape[1]), reverse=True)
    big = dfs[0] if dfs else pd.DataFrame()
    return flatten_columns(big)

def _ensure_ecr_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ecr_rank" not in out.columns:
        # build rank from 'ecr' or 'avg' if present; else from current order
        if "ecr" in out.columns:
            out["ecr_rank"] = pd.to_numeric(out["ecr"], errors="coerce").rank(method="min", ascending=True)
        elif "avg" in out.columns:
            out["ecr_rank"] = pd.to_numeric(out["avg"], errors="coerce").rank(method="min", ascending=True)
        else:
            out["ecr_rank"] = range(1, len(out) + 1)
    return out

def _normalize_ecr_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # rename common headers from the DOM table
    name_col = next((c for c in out.columns if re.search(r"\bplayer\b|\bname\b", c, re.I)), None)
    team_col = next((c for c in out.columns if re.search(r"\bteam\b", c, re.I)), None)
    pos_col  = next((c for c in out.columns if re.fullmatch(r"pos(ition)?", c, flags=re.I)), None)
    rk_col   = next((c for c in out.columns if re.fullmatch(r"rk|rank", c, flags=re.I)), None)
    ecr_col  = next((c for c in out.columns if re.fullmatch(r"ecr", c, flags=re.I)), None)

    if name_col and name_col != "player_name": out.rename(columns={name_col: "player_name"}, inplace=True)
    if team_col and team_col != "team":        out.rename(columns={team_col: "team"}, inplace=True)
    if pos_col and pos_col != "pos":           out.rename(columns={pos_col: "pos"}, inplace=True)
    if rk_col and rk_col != "ecr_rank":        out.rename(columns={rk_col: "ecr_rank"}, inplace=True)
    if ecr_col and ecr_col != "ecr":           out.rename(columns={ecr_col: "ecr"}, inplace=True)

    out = _ensure_ecr_rank(out)
    # numeric coercion
    for c in ("ecr_rank", "ecr"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def fetch_ecr_overall(year: int, scoring: str, sess: requests.Session) -> Tuple[pd.DataFrame, str]:
    """Fetch ONLY from the cheatsheets page for the specified scoring; return full 'overall' rankings."""
    url = _ecr_url(scoring, year)
    resp = sess.get(url, timeout=30, allow_redirects=True)
    if "account/login" in resp.url:
        return pd.DataFrame(), url

    # 1) JSON-first (usually the complete rankings)
    payload = _try_extract_ecr_json(resp.text)
    if payload:
        df = _ecr_json_to_df(payload)
        if not df.empty and len(df) >= 15:
            df = _normalize_ecr_columns(df)
            return df, url

    # 2) DOM table (id="ranking-table")
    df_dom = _read_rank_table_from_dom(resp.text)
    if not df_dom.empty and len(df_dom) >= 15:
        df_dom = _normalize_ecr_columns(df_dom)
        return df_dom, url

    # 3) Generic fallback on that same page (largest Rank+Player table)
    try:
        tables = read_tables(resp.text)
    except ValueError:
        tables = []
    best = None; score = (-1, -1)
    for t in tables:
        t2 = flatten_columns(t)
        up = [str(c).upper() for c in t2.columns]
        if any(("PLAYER" in c) or ("NAME" in c) for c in up) and any(("RK" == c) or ("RANK" in c) for c in up):
            key = (len(t2), t2.shape[1])
            if key > score:
                best, score = t2, key
    if best is not None and len(best) >= 15:
        best = _normalize_ecr_columns(best)
        return best, url

    return pd.DataFrame(), url

def _filter_pos(df_overall: pd.DataFrame, pos: str) -> pd.DataFrame:
    """Filter the overall ECR frame to a specific position and compute positional rank."""
    out = df_overall.copy()
    if "pos" not in out.columns:
        return pd.DataFrame()
    want = pos.upper()
    if want == "DST":
        patt = r"(DST|DEF|D/?ST)"
    elif want == "K":
        patt = r"^K$"
    elif want == "QB":
        patt = r"^QB$"
    elif want == "RB":
        patt = r"^RB$"
    elif want == "WR":
        patt = r"^WR$"
    elif want == "TE":
        patt = r"^TE$"
    else:
        return pd.DataFrame()
    sub = out[out["pos"].astype(str).str.upper().str.contains(patt, regex=True)].copy()
    if sub.empty:
        return sub
    # positional rank (1 = best within position)
    if "ecr_rank" in sub.columns:
        sub["ecr_pos_rank"] = sub["ecr_rank"].rank(method="min", ascending=True).astype(int)
    return sub

def harvest_ecr(years=YEARS, scorings=SCORINGS, positions=("overall","qb","rb","wr","te","k","dst"), out_dir="fp_ecr"):
    """
    Fetch overall ECR from ONLY the two cheatsheets endpoints you specified,
    then write per-position CSVs by filtering the overall table.
    """
    os.makedirs(out_dir, exist_ok=True)
    sess = session_with_retry()
    for y in years:
        for s in scorings:
            try:
                time.sleep(random.uniform(*DEFAULT_SLEEP))
                overall, used = fetch_ecr_overall(y, s, sess)
                if overall.empty:
                    print(f"[MISS] ECR {y} {s} overall: no table ({used})")
                    continue

                # annotate and save overall
                df_overall = overall.copy()
                df_overall["season"] = y
                df_overall["scoring"] = s.lower()
                df_overall["pos_slug"] = "OVERALL"
                df_overall["source_url"] = used
                # keep common columns first
                core = [c for c in ["player_name","team","pos","ecr_rank","ecr","season","scoring","pos_slug","source_url"] if c in df_overall.columns]
                other = [c for c in df_overall.columns if c not in core]
                df_overall = df_overall[core + other]
                fname = f"fp_ecr_{y}_{s.lower()}_overall.csv"
                df_overall.to_csv(os.path.join(out_dir, fname), index=False)
                print(f"[OK]   ECR {y} {s} overall -> {fname}")

                # derive and save each position from the same overall table
                for p in [p for p in positions if p != "overall"]:
                    sub = _filter_pos(df_overall, p)
                    if sub.empty:
                        print(f"[MISS] ECR {y} {s} {p}: filter produced 0 rows")
                        continue
                    out = sub.copy()
                    out["season"] = y
                    out["scoring"] = s.lower()
                    out["pos_slug"] = p.upper()
                    out["source_url"] = used
                    core = [c for c in ["player_name","team","pos","ecr_rank","ecr","ecr_pos_rank",
                                        "season","scoring","pos_slug","source_url"] if c in out.columns]
                    other = [c for c in out.columns if c not in core]
                    out = out[core + other]
                    fname = f"fp_ecr_{y}_{s.lower()}_{p}.csv"
                    out.to_csv(os.path.join(out_dir, fname), index=False)
                    print(f"[OK]   ECR {y} {s} {p} -> {fname}")

            except Exception as e:
                print(f"[ERR]  ECR {y} {s}: {e}")

# =========================================================
# Optional: simple ADP+ECR join (kept for convenience)
# =========================================================

def join_adp_ecr(year: int, scoring: str, pos: str,
                 adp_dir="fp_adp", ecr_dir="fp_ecr", out_dir="fp_join") -> str:
    os.makedirs(out_dir, exist_ok=True)
    adp_path = os.path.join(adp_dir, f"fp_adp_{year}_{scoring.lower()}_{pos}.csv")
    ecr_path = os.path.join(ecr_dir, f"fp_ecr_{year}_{scoring.lower()}_{pos}.csv")
    if not (os.path.exists(adp_path) and os.path.exists(ecr_path)):
        return ""

    adp = pd.read_csv(adp_path)
    ecr = pd.read_csv(ecr_path)

    def clean_name(s):
        if pd.isna(s): return s
        s = re.sub(r"\s+\(.*?\)$", "", str(s))  # strip trailing (Team) variants if any
        return re.sub(r"\s+", " ", s).strip().lower()

    for df in (adp, ecr):
        if "player_name" in df.columns:
            df["player_key"] = df["player_name"].map(clean_name)
        else:
            df["player_key"] = pd.NA

    merged = pd.merge(ecr, adp, on=["player_key"], how="inner", suffixes=("_ecr", "_adp"))
    out_path = os.path.join(out_dir, f"fp_adp_ecr_{year}_{scoring.lower()}_{pos}.csv")
    merged.to_csv(out_path, index=False)
    return out_path

# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    # Uncomment if you also want ADP:
    # harvest_adp()
    harvest_ecr()

    # Example join:
    # for y in YEARS:
    #     for s in SCORINGS:
    #         for p in ["overall", "wr"]:
    #             path = join_adp_ecr(y, s, p)
    #             if path:
    #                 print(f"[OK] JOIN {y} {s} {p} -> {path}")
