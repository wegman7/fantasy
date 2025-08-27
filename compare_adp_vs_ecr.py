import os, re, glob
import numpy as np
import pandas as pd

# ------------ config ------------
YEARS    = list(range(2015, 2025))
SCORINGS = ["ppr", "half"]
POSITIONS = ["qb", "rb", "wr", "te", "k", "dst"]   # you can add "overall" if you want, but we compare per-position

DIR_ECR   = "fp_ecr"
DIR_ADP   = "fp_adp"
DIR_STATS = "fp_season_stats"
OUT_DIR   = "fp_metrics"

TOPX_DEFAULTS = {"QB": 12, "RB": 24, "WR": 36, "TE": 12, "K": 12, "DST": 12}

# Team name normalization in case DST joins need full-name<->abbr (best effort)
TEAM_ABBR = {
    # Common/official names -> ABBR
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
    "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
    "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC",
    "LAS VEGAS RAIDERS":"LV","OAKLAND RAIDERS":"OAK","LOS ANGELES CHARGERS":"LAC","SAN DIEGO CHARGERS":"SD",
    "LOS ANGELES RAMS":"LAR","ST. LOUIS RAMS":"STL","MIAMI DOLPHINS":"MIA","MINNESOTA VIKINGS":"MIN",
    "NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG","NEW YORK JETS":"NYJ",
    "PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT","SAN FRANCISCO 49ERS":"SF","SEATTLE SEAHAWKS":"SEA",
    "TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WAS","WASHINGTON FOOTBALL TEAM":"WAS",
    "WASHINGTON REDSKINS":"WAS"
}
# Also accept abbreviations as-is
for ab in list(set(TEAM_ABBR.values())):
    TEAM_ABBR[ab] = ab

def _guess_player_name_column(df: pd.DataFrame) -> str | None:
    # 1) header contains 'player' or 'name'
    for c in df.columns:
        if re.search(r"\b(player|name)\b", str(c), re.I):
            return c
    # 2) heuristic: first object column that looks like "Firstname Lastname …"
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in obj_cols:
        s = df[c].astype(str)
        # lots of spaces, not mostly 2–4 letter all-caps (team codes)
        has_spaces = (s.str.contains(r"\s").mean() > 0.6)
        looks_like_team = (s.str.match(r"^[A-Z]{2,4}$").mean() > 0.6)
        if has_spaces and not looks_like_team:
            return c
    return None

def _ensure_player_name_and_key(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'player_name' and 'player_key' exist, deriving them if needed."""
    out = df.copy()
    if "player_name" not in out.columns:
        cand = _guess_player_name_column(out)
        if cand:
            out.rename(columns={cand: "player_name"}, inplace=True)

    if "player_name" in out.columns and "player_key" not in out.columns:
        out["player_key"] = out["player_name"].astype(str).map(clean_name)

    return out

def _require_player_key_or_skip(df: pd.DataFrame, label: str, year: int, scoring: str, pos: str) -> pd.DataFrame:
    """Last-chance attempt; print debug and return empty if we still can't build a key."""
    out = _ensure_player_name_and_key(df)
    if "player_key" not in out.columns:
        print(f"[SKIP] {year} {scoring} {pos}: cannot find a player name column in {label}. "
              f"Columns={list(df.columns)[:12]}")
        return pd.DataFrame()
    return out

def _coerce_points_columns(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Add fantasy_points / fantasy_points_per_game if the CSV used alt headers (e.g., MISC_FPTS)."""
    df = stats_df.copy()
    # Build a normalized lookup (strip non-letters, uppercase)
    norm_map = {c: re.sub(r"[^A-Za-z]", "", str(c)).upper() for c in df.columns}

    # Find total points
    fpts_col = None
    for c, n in norm_map.items():
        if ("FPTS" == n) or ("FANTASYPOINTS" in n) or (n.endswith("FPTS")):  # catches MISC_FPTS, etc.
            fpts_col = c
            break
    if fpts_col and "fantasy_points" not in df.columns:
        df = df.rename(columns={fpts_col: "fantasy_points"})

    # Find points per game
    fptsg_col = None
    for c, n in norm_map.items():
        if ("FPTSG" == n) or ("FANTASYPOINTSPERGAME" in n) or (n.endswith("FPTSG")):
            fptsg_col = c
            break
    if fptsg_col and "fantasy_points_per_game" not in df.columns:
        df = df.rename(columns={fptsg_col: "fantasy_points_per_game"})

    return df

def clean_name(s: str) -> str:
    """Normalize player names for joins."""
    if pd.isna(s): return ""
    s = str(s)
    # strip team suffixes like " (FA)" if present
    s = re.sub(r"\s+\(.*?\)$", "", s)
    # remove D/ST suffix in names like "Patriots D/ST"
    s = re.sub(r"\s+D/?ST$", "", s, flags=re.I)
    # remove Jr., Sr., III, II (light touch)
    s = re.sub(r"\b(JR\.?|SR\.?|III|II)\b", "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z '\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)

    # ensure name + key
    df = _ensure_player_name_and_key(df)

    # normalize position + scoring
    if "pos_slug" in df.columns:
        df["pos_slug"] = df["pos_slug"].astype(str).str.upper()
    if "scoring" in df.columns:
        df["scoring"] = df["scoring"].astype(str).str.lower()

    # normalize team to ABBR for joins (DST especially)
    if "team" in df.columns and "team_norm" not in df.columns:
        df["team_norm"] = df["team"].astype(str).str.upper().map(lambda x: TEAM_ABBR.get(x, x))

    return df

def ensure_ecr_rank(df: pd.DataFrame) -> pd.DataFrame:
    if "ecr_rank" in df.columns and df["ecr_rank"].notna().any():
        return df
    # If "ecr" exists, lower is better -> rank ascending
    if "ecr" in df.columns:
        df["ecr_rank"] = df["ecr"].rank(method="min", ascending=True)
    elif "rk" in df.columns:
        df["ecr_rank"] = pd.to_numeric(df["rk"], errors="coerce")
    else:
        # last resort: make a dense rank on the row order
        df["ecr_rank"] = np.arange(1, len(df) + 1)
    return df

def add_adp_ranks(df_adp: pd.DataFrame) -> pd.DataFrame:
    """Create rank columns from ADP values (smaller ADP -> earlier pick -> better)."""
    out = df_adp.copy()
    for col in ["adp_espn","adp_yahoo","adp_avg"]:
        if col in out.columns:
            out[col+"_rank"] = out[col].rank(method="min", ascending=True)
    return out

def final_ranks_from_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
    df = _coerce_points_columns(stats_df)          # <— NEW
    if df.empty:
        return df
    if "fantasy_points" not in df.columns:
        raise ValueError("stats file missing 'fantasy_points' (after coercion)")

    if "pos_slug" not in df.columns:
        if "pos" in df.columns:
            df["pos_slug"] = df["pos"].astype(str).str.upper()
        else:
            raise ValueError("stats file missing 'pos_slug'/'pos'")

    df["final_pos_rank"] = df.groupby("pos_slug")["fantasy_points"].rank(method="min", ascending=False)
    return df


def join_ecr_adp_final(year: int, scoring: str, pos: str):
    posU = pos.upper()
    ecr_path   = os.path.join(DIR_ECR,   f"fp_ecr_{year}_{scoring}_{pos}.csv")
    adp_path   = os.path.join(DIR_ADP,   f"fp_adp_{year}_{scoring}_{pos}.csv")
    stats_path = os.path.join(DIR_STATS, f"fp_stats_{year}_{scoring}_{pos}.csv")

    ecr  = load_csv(ecr_path)
    adp  = load_csv(adp_path)
    fin  = load_csv(stats_path)

    # last-chance enforcement (and friendly debug) before we touch 'player_key'
    ecr = _require_player_key_or_skip(ecr,  "ECR",  year, scoring, pos)
    adp = _require_player_key_or_skip(adp,  "ADP",  year, scoring, pos)
    fin = _require_player_key_or_skip(fin,  "STATS",year, scoring, pos)
    if ecr.empty or adp.empty or fin.empty:
        return pd.DataFrame()

    ecr  = ensure_ecr_rank(ecr)
    adp  = add_adp_ranks(adp)
    fin  = final_ranks_from_stats(fin)

    # Build keys:
    # - For DST: prefer team-based join; for players: name+pos
    if posU == "DST":
        # unchanged
        ecr_key = ecr[["team_norm","ecr_rank","player_key","player_name"]].rename(columns={"team_norm":"key"})
        adp_key = adp[["team_norm","adp_espn","adp_yahoo","adp_avg",
                    "adp_espn_rank","adp_yahoo_rank","adp_avg_rank",
                    "player_key","player_name"]].rename(columns={"team_norm":"key"})
        fin_key = fin[["team_norm","fantasy_points","final_pos_rank"]].rename(columns={"team_norm":"key"})
    else:
        # build the join key
        ecr["key"] = ecr["player_key"] + "|" + posU
        adp["key"] = adp["player_key"] + "|" + posU
        fin["key"] = fin["player_key"] + "|" + posU

    # only include columns that actually exist
    ecr_cols = ["key","ecr_rank","player_name"]
    if "team_norm" in ecr.columns: ecr_cols.append("team_norm")
    ecr_key = ecr[ecr_cols]

    adp_cols = ["key","adp_espn","adp_yahoo","adp_avg",
                "adp_espn_rank","adp_yahoo_rank","adp_avg_rank","player_name"]
    if "team_norm" in adp.columns: adp_cols.append("team_norm")
    adp_key = adp[adp_cols]

    fin_key = fin[["key","fantasy_points","final_pos_rank"]]

    # Merge
    df = pd.merge(ecr_key, adp_key, on=list(set(ecr_key.columns) & set(adp_key.columns)), how="inner")
    df = pd.merge(df, fin_key, on=list(set(df.columns) & set(fin_key.columns)), how="inner")

    # add context
    df["season"] = year
    df["scoring"] = scoring
    df["pos_slug"] = posU
    # prune to useful columns
    keep = ["season","scoring","pos_slug","player_name","team_norm","ecr_rank",
            "adp_espn","adp_yahoo","adp_avg",
            "adp_espn_rank","adp_yahoo_rank","adp_avg_rank",
            "fantasy_points","final_pos_rank"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].dropna(subset=["final_pos_rank"])  # require ground truth

def evaluate_one(df: pd.DataFrame, posU: str, topX_map=TOPX_DEFAULTS):
    """Compute metrics for each predictor on this merged DF."""
    if df.empty:
        return pd.DataFrame()
    # predictors: ECR rank, ADP ranks
    preds = {
        "ECR": "ecr_rank",
        "ADP_ESPN": "adp_espn_rank" if "adp_espn_rank" in df.columns else None,
        "ADP_YAHOO": "adp_yahoo_rank" if "adp_yahoo_rank" in df.columns else None,
        "ADP_AVG": "adp_avg_rank" if "adp_avg_rank" in df.columns else None,
    }
    preds = {k:v for k,v in preds.items() if v}

    topX = topX_map.get(posU, 12)

    out = []
    for name, col in preds.items():
        # rank correlation
        rho = df[col].corr(df["final_pos_rank"], method="spearman")
        # errors
        err = df[col] - df["final_pos_rank"]
        mae = err.abs().mean()
        rmse = np.sqrt((err**2).mean())

        # top-X hit rate: among top-X predicted by this predictor, how many finish top-X?
        df_sorted = df.sort_values(col, ascending=True)
        pred_top = set(df_sorted.head(topX).index)
        final_top = set(df.sort_values("final_pos_rank").head(topX).index)
        hit_rate = len(pred_top & final_top) / float(topX)

        out.append({
            "season": int(df["season"].iloc[0]),
            "scoring": df["scoring"].iloc[0],
            "pos_slug": posU,
            "model": name,
            "spearman_r": rho,
            "mae_rank": mae,
            "rmse_rank": rmse,
            "topX": topX,
            "hit_rate_topX": hit_rate,
            "n_players": len(df)
        })
    return pd.DataFrame(out)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_metrics = []
    for y in YEARS:
        for s in SCORINGS:
            for p in POSITIONS:
                try:
                    merged = join_ecr_adp_final(y, s, p)
                    if merged.empty:
                        print(f"[SKIP] {y} {s} {p} (no merged rows)")
                        continue
                    met = evaluate_one(merged, p.upper())
                    if not met.empty:
                        all_metrics.append(met)
                        # optional: write per-slice merged table for auditing
                        merged.to_csv(os.path.join(OUT_DIR, f"merged_{y}_{s}_{p}.csv"), index=False)
                        print(f"[OK] {y} {s} {p}: {len(merged)} rows, metrics computed")
                except Exception as e:
                    print(f"[ERR] {y} {s} {p}: {e}")
    if all_metrics:
        metrics = pd.concat(all_metrics, ignore_index=True)
        metrics = metrics.sort_values(["season","scoring","pos_slug","model"])
        metrics.to_csv(os.path.join(OUT_DIR, "metrics_ecr_vs_adp.csv"), index=False)
        print(f"wrote {os.path.join(OUT_DIR, 'metrics_ecr_vs_adp.csv')} ({len(metrics)} rows)")
    else:
        print("No metrics produced. Check that your input CSVs exist.")

if __name__ == "__main__":
    main()
