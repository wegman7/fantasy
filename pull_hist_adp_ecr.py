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
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.fantasypros.com/nfl/",
        "Upgrade-Insecure-Requests": "1",
        "Connection": "keep-alive",
    })
    if Retry is not None:
        r = Retry(
            total=5, backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
        )
        s.mount("https://", HTTPAdapter(max_retries=r))
    else:
        s.mount("https://", HTTPAdapter())
    return s


def read_tables(html_text: str) -> List[pd.DataFrame]:
    """Try pandas directly; if it fails, extract table#ranking-table then parse."""
    # 1) try global parse
    try:
        return pd.read_html(io.StringIO(html_text), displayed_only=False)
    except ValueError:
        pass

    # 2) narrow to the main rankings table
    try:
        from bs4 import BeautifulSoup
    except Exception:
        # If bs4 isn't installed, just propagate the original "no tables" behavior
        raise

    soup = BeautifulSoup(html_text, "lxml")
    table = soup.select_one("table#ranking-table")
    if table is None:
        # try a slightly looser selector used on that page
        table = soup.select_one('table.table.player-table')
    if table is None:
        raise ValueError("No tables found (ranking-table not present)")

    # Now parse just that table
    return pd.read_html(io.StringIO(str(table)), displayed_only=False)


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

from urllib.parse import urljoin

def _abs(url: str) -> str:
    return url if url.startswith("http") else urljoin(BASE, url)


def _build_ecr_attempt_urls(year: int, scoring: str, pos: str) -> list[str]:
    """Build only the overall cheatsheet URL (more stable), we will filter by pos later."""
    scoring = scoring.upper()
    if scoring == "PPR":
        path = "/nfl/rankings/ppr-cheatsheets.php"
    elif scoring in ("HALF", "HALF_PPR", "HALF-POINT-PPR"):
        path = "/nfl/rankings/half-point-ppr-cheatsheets.php"
    else:
        # fallback to PPR page if something unexpected is passed
        path = "/nfl/rankings/ppr-cheatsheets.php"
    return [f"{BASE}{path}?year={year}"]


def _pick_ecr_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Pick the widest table that looks like the cheatsheet:
    must have Player column and at least one of Rank/Rk/ECR/Tier/Avg.
    """
    best, best_cols = None, -1
    for df in tables:
        df2 = flatten_columns(df)
        cols_up = [str(c).upper() for c in df2.columns]
        has_player = any("PLAYER" in c or "NAME" in c for c in cols_up)

        # Cheatsheet sometimes has TIER + AVG instead of RANK/ECR
        has_indicator = any(
            ("RANK" in c) or (c == "RK") or ("ECR" in c) or ("TIER" in c) or (c == "AVG")
            for c in cols_up
        )

        if has_player and has_indicator:
            if df2.shape[1] > best_cols:
                best, best_cols = df2, df2.shape[1]
    return best if best is not None else pd.DataFrame()


from bs4 import BeautifulSoup

def _extract_cheatsheet_table_html(page_html: str) -> str | None:
    """
    Try (a) direct table#ranking-table, (b) any table with .player-table,
    (c) follow data-url to fetch server-rendered table/rows.
    Returns HTML for a single <table> or None.
    """
    soup = BeautifulSoup(page_html, "lxml")

    # (a) direct table
    table = soup.select_one("table#ranking-table")
    if table:
        return str(table)

    # (b) looser selector
    table = soup.select_one("table.table.player-table")
    if table:
        return str(table)

    # (c) Ajax endpoint referenced on the page
    # Look for common attributes used by FantasyPros pages:
    #   data-url on the table or on a nearby container
    data_url = None
    cand = soup.select_one("[data-url]")
    if cand and cand.has_attr("data-url"):
        data_url = cand["data-url"]

    # Some pages use data-table-url or data-src
    if not data_url:
        cand = soup.select_one("[data-table-url]")
        if cand and cand.has_attr("data-table-url"):
            data_url = cand["data-table-url"]
    if not data_url:
        cand = soup.select_one("[data-src]")
        if cand and cand.has_attr("data-src"):
            data_url = cand["data-src"]

    if not data_url:
        return None

    # Ensure absolute URL
    ajax_url = data_url if data_url.startswith("http") else f"{BASE}{data_url}"
    # Pull the Ajax content
    sess = session_with_retry()
    r = sess.get(ajax_url, timeout=25)
    if r.status_code >= 400 or "account/login" in r.url:
        return None

    # Ajax may return a <table> or just <tr> rows. Normalize to a table string.
    html = r.text
    ajax_soup = BeautifulSoup(html, "lxml")
    table = ajax_soup.select_one("table") or ajax_soup.select_one("tbody")
    if table and table.name == "tbody":
        # wrap rows into a minimal table so pandas can parse it
        return f"<table><thead></thead>{str(table)}</table>"
    if table:
        return str(table)
    # As a last resort, if the response looks like rows, wrap them
    if "<tr" in html:
        return f"<table><thead></thead><tbody>{html}</tbody></table>"
    return None


from bs4 import BeautifulSoup

def _find_csv_link(page_html: str) -> str | None:
    """
    FantasyPros cheatsheet pages expose a 'Download CSV' link.
    We search anchors for hrefs that look like CSV exports.
    """
    soup = BeautifulSoup(page_html, "lxml")

    # Prefer explicit buttons/links that look like exports
    # Heuristics: href contains 'download' or 'export' and 'csv'
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = (a.get_text() or "").strip().lower()
        hlow = href.lower()
        if ("csv" in hlow) and ("download" in hlow or "export" in hlow or "download" in text or "csv" in text):
            return _abs(href)

    # Secondary: any link that ends with .csv
    for a in soup.find_all("a", href=True):
        hlow = a["href"].lower()
        if hlow.endswith(".csv"):
            return _abs(a["href"])

    return None


def _parse_cheatsheet_table_html(page_html: str) -> pd.DataFrame:
    """
    Fallback: try to parse the table directly from the HTML (if present).
    Works only when rows are server-rendered (sometimes they aren't).
    """
    try:
        # Global parse
        tables = pd.read_html(io.StringIO(page_html), displayed_only=False)
    except ValueError:
        tables = []

    def pick(tbls):
        best, best_cols = None, -1
        for df in tbls:
            df2 = flatten_columns(df)
            cols_up = [str(c).upper() for c in df2.columns]
            has_player = any(("PLAYER" in c) or ("NAME" in c) for c in cols_up)
            has_indicator = any(("RANK" in c) or (c == "RK") or ("ECR" in c) or ("TIER" in c) or (c == "AVG") for c in cols_up)
            if has_player and has_indicator and df2.shape[1] > best_cols:
                best, best_cols = df2, df2.shape[1]
        return best if best is not None else pd.DataFrame()

    df = pick(tables)
    if not df.empty:
        return df

    # Target the specific table if global read didn't find it
    soup = BeautifulSoup(page_html, "lxml")
    table = soup.select_one("table#ranking-table") or soup.select_one("table.table.player-table")
    if not table:
        return pd.DataFrame()

    try:
        tbls2 = pd.read_html(io.StringIO(str(table)), displayed_only=False)
    except ValueError:
        return pd.DataFrame()

    return pick(tbls2)


def fetch_ecr(year: int, scoring: str, pos: str, sess: requests.Session) -> tuple[pd.DataFrame, str]:
    """
    1) Load cheatsheet page
    2) Find and download CSV export (preferred)
    3) Fallback to parsing the HTML table if no CSV link found
    """
    for url in _build_ecr_attempt_urls(year, scoring, pos):
        resp = sess.get(url, timeout=30, allow_redirects=True)
        if resp.status_code >= 400 or "account/login" in resp.url:
            continue
        html = resp.text

        # --- Preferred path: CSV export ---
        csv_url = _find_csv_link(html)
        if csv_url:
            r2 = sess.get(csv_url, timeout=30)
            if r2.status_code < 400 and r2.content:
                # Many FP CSVs are straightforward; use read_csv directly
                df_csv = pd.read_csv(io.StringIO(r2.text))
                if not df_csv.empty:
                    return df_csv, csv_url  # return the CSV source

        # --- Fallback: HTML table parsing ---
        df_tbl = _parse_cheatsheet_table_html(html)
        if not df_tbl.empty:
            return df_tbl, url

    return pd.DataFrame(), ""




def normalize_ecr_df(df: pd.DataFrame, pos: str, scoring: str, year: int, url: str) -> pd.DataFrame:
    out = flatten_columns(df)

    # Column discovery
    name_col = next((c for c in out.columns if re.search(r"\bplayer\b|\bname\b", c, re.I)), None)
    team_col = next((c for c in out.columns if re.fullmatch(r"team(\s*\(bye\))?", c, flags=re.I)), None)
    pos_col  = next((c for c in out.columns if re.fullmatch(r"pos(ition)?", c, flags=re.I)), None)
    rk_col   = next((c for c in out.columns if re.fullmatch(r"rk|rank", c, flags=re.I)), None)
    ecr_col  = next((c for c in out.columns if re.fullmatch(r"ecr", c, flags=re.I)), None)
    avg_col  = next((c for c in out.columns if re.fullmatch(r"avg", c, flags=re.I)), None)

    # Rename to our canonical schema
    ren = {}
    if name_col and name_col != "player_name": ren[name_col] = "player_name"
    if team_col and team_col != "team":        ren[team_col] = "team"
    if pos_col  and pos_col  != "pos":         ren[pos_col]  = "pos"
    if rk_col   and rk_col   != "ecr_rank":    ren[rk_col]   = "ecr_rank"
    if ecr_col  and ecr_col  != "ecr":         ren[ecr_col]  = "ecr"
    if avg_col and (not ecr_col):              ren[avg_col]  = "ecr"  # cheatsheet's AVG ≈ consensus rank/score

    out.rename(columns=ren, inplace=True)

    # Derive ecr_rank if missing: use 1..N order
    if "ecr_rank" not in out.columns:
        out["ecr_rank"] = range(1, len(out) + 1)

    # Coerce numerics where present
    for c in ("ecr_rank", "ecr"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Basic fields
    out["season"]   = year
    out["scoring"]  = scoring.lower()
    out["pos_slug"] = pos.upper()
    out["source_url"] = url

    # Keep only reasonable columns, in a stable order
    core = [c for c in ["player_name","team","pos","ecr_rank","ecr","season","scoring","pos_slug","source_url"] if c in out.columns]
    other = [c for c in out.columns if c not in core]
    out = out[core + other]

    return out


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

                    # If caller asked for a specific position, filter by it when we have a POS column.
                    # Cheatsheet POS values are like 'QB','RB','WR','TE','K','DST'.
                    if p.lower() != "overall" and "pos" in df.columns:
                        df = df[df["pos"].str.upper().eq(p.upper())].reset_index(drop=True)

                    fname = f"fp_ecr_{y}_{s.lower()}_{p}.csv"
                    df.to_csv(os.path.join(out_dir, fname), index=False)
                    print(f"[OK]   ECR {y} {s} {p} -> {fname}")
                except Exception as e:
                    print(f"[ERR]  ECR {y} {s} {p}: {e}")
                    print(e)


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
