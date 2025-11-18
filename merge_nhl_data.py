# merge_nhl_data.py — Cloud-Safe RW + MoneyPuck Merge (L10/L20)
# Uses MoneyPuck recency (L10 + L20) for skaters and (where possible) goalies.
# 100% portable for Render, Streamlit Cloud, Replit, etc.

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from io import StringIO
import time

# ---------------------------------------------------------
# CLOUD-SAFE PATHS
# ---------------------------------------------------------
DATA_DIR = Path(__file__).parent
RW_FILE = DATA_DIR / "rw-nhl-player-pool.xlsx"
MERGED_OUT = DATA_DIR / "merged_nhl_player_pool.csv"

# Cache directory for fallback when MP is down
CACHE_DIR = DATA_DIR / "moneypuck_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Utility — build cache path
# ---------------------------------------------------------
def _cache_path(label: str):
    safe = label.replace(" ", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}.csv"


# ---------------------------------------------------------
# MoneyPuck CSV downloader (Retry + Cache + UA)
# ---------------------------------------------------------
def _load_mp_csv(url: str, label: str) -> pd.DataFrame:
    """
    Extremely robust MoneyPuck CSV loader.
    - Adds User-Agent (MP blocks default Python requests)
    - Retries 3 times
    - Falls back to cached CSV if available
    """

    print(f"[INFO] Fetching MoneyPuck {label} → {url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; x86_64)",
        "Accept": "text/csv,*/*;q=0.9",
    }

    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code == 200:
                text = r.text.strip()
                if len(text) < 30:
                    print(f"[WARN] MoneyPuck returned empty CSV for {label}, retrying...")
                else:
                    df = pd.read_csv(StringIO(text))
                    df.to_csv(_cache_path(label), index=False)
                    print(f"[OK] Loaded {label}: {len(df)} rows")
                    return df
            else:
                print(f"[WARN] MP HTTP {r.status_code} for {label}, retry {attempt+1}/3...")
        except Exception as e:
            print(f"[WARN] Attempt {attempt+1} failed for {label}: {e}")

        time.sleep(1.5)

    # Fallback: read cached copy
    cache_file = _cache_path(label)
    if cache_file.exists():
        print(f"[FALLBACK] Using cached MoneyPuck data for {label}")
        return pd.read_csv(cache_file)

    raise RuntimeError(f"[ERROR] MoneyPuck failed for {label} and no cache exists.")


# ---------------------------------------------------------
# MoneyPuck endpoints (2025)
# ---------------------------------------------------------
def load_skaters_l10():
    return _load_mp_csv(
        "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/skaters_10.csv",
        "skaters_L10",
    )


def load_skaters_l20():
    return _load_mp_csv(
        "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/skaters_20.csv",
        "skaters_L20",
    )


def load_goalies_l10():
    return _load_mp_csv(
        "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies_10.csv",
        "goalies_L10",
    )


def load_goalies_l20():
    return _load_mp_csv(
        "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies_20.csv",
        "goalies_L20",
    )


# ---------------------------------------------------------
# Normalize MoneyPuck Name → Rotowire Format
# ("Matthews, Auston" → "Auston Matthews")
# ---------------------------------------------------------
def _normalize_mp_name(raw):
    raw = str(raw).strip()
    if "," in raw:
        last, first = raw.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return raw


# ---------------------------------------------------------
# Load Rotowire Excel
# ---------------------------------------------------------
def load_rotowire(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Rotowire file missing: {path}")

    print(f"[INFO] Loading Rotowire → {path}")
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Normalized RW player column
    if "PLAYER" not in df.columns:
        raise RuntimeError("Rotowire file must contain a PLAYER column.")

    df["PLAYER"] = df["PLAYER"].astype(str).str.strip()
    print(f"[OK] Rotowire rows: {len(df)}")
    return df


# ---------------------------------------------------------
# Build recency features from L10/L20
# ---------------------------------------------------------
def add_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive recency-aware fields expected by nhl_projection_engine.py and nhl_line_model.py:
      - xG_per60_recency
      - SOG_per60_recency
      - xGA_pg_recency_goalie (if goalie GA data is present)
      - xGA_pg_recency (team-level, from goalie recency)
    We treat the base (no suffix) MP columns as L10, and *_L20 as L20.
    """
    df = df.copy()

    # ---------- Skater recency: xG_per60_recency / SOG_per60_recency ----------
    xg10 = pd.to_numeric(df.get("I_F_xGoals", np.nan), errors="coerce")
    xg20 = pd.to_numeric(df.get("I_F_xGoals_L20", np.nan), errors="coerce")

    sog10 = pd.to_numeric(df.get("I_F_shotsOnGoal", np.nan), errors="coerce")
    sog20 = pd.to_numeric(df.get("I_F_shotsOnGoal_L20", np.nan), errors="coerce")

    toi10 = pd.to_numeric(df.get("icetime", np.nan), errors="coerce")       # seconds (L10)
    toi20 = pd.to_numeric(df.get("icetime_L20", np.nan), errors="coerce")   # seconds (L20)

    with np.errstate(divide="ignore", invalid="ignore"):
        xg60_10 = np.where(toi10 > 0, xg10 * 3600.0 / toi10, np.nan)
        xg60_20 = np.where(toi20 > 0, xg20 * 3600.0 / toi20, np.nan)
        sog60_10 = np.where(toi10 > 0, sog10 * 3600.0 / toi10, np.nan)
        sog60_20 = np.where(toi20 > 0, sog20 * 3600.0 / toi20, np.nan)

    xg60_10 = pd.Series(xg60_10, index=df.index)
    xg60_20 = pd.Series(xg60_20, index=df.index)
    sog60_10 = pd.Series(sog60_10, index=df.index)
    sog60_20 = pd.Series(sog60_20, index=df.index)

    # Blend with heavier weight on L10 when both available
    xg_rec = xg60_10.where(xg60_10.notna(), xg60_20)
    mask_both_xg = xg60_10.notna() & xg60_20.notna()
    xg_rec[mask_both_xg] = 0.65 * xg60_10[mask_both_xg] + 0.35 * xg60_20[mask_both_xg]

    sog_rec = sog60_10.where(sog60_10.notna(), sog60_20)
    mask_both_sog = sog60_10.notna() & sog60_20.notna()
    sog_rec[mask_both_sog] = 0.65 * sog60_10[mask_both_sog] + 0.35 * sog60_20[mask_both_sog]

    # Fallback to season per-60 if recency missing
    xg60_season = pd.to_numeric(df.get("xG_per60", np.nan), errors="coerce")
    sog60_season = pd.to_numeric(
        df.get("SOG_per60", df.get("shots_on_goal_per60", np.nan)),
        errors="coerce",
    )

    xg_rec = xg_rec.where(xg_rec.notna(), xg60_season)
    sog_rec = sog_rec.where(sog_rec.notna(), sog60_season)

    df["xG_per60_recency"] = xg_rec.fillna(0.0)
    df["SOG_per60_recency"] = sog_rec.fillna(0.0)

    # ---------- Goalie / team defensive recency (if we have GA data) ----------
    # We look for typical MoneyPuck goalie columns with "_G10"/"_G20" suffix.
    ga10_col = next(
        (c for c in df.columns if "goalsAgainst" in c and c.endswith("_G10")),
        None,
    )
    gp10_col = next(
        (
            c
            for c in df.columns
            if (("gamesPlayed" in c) or ("games_played" in c)) and c.endswith("_G10")
        ),
        None,
    )
    ga20_col = next(
        (c for c in df.columns if "goalsAgainst" in c and c.endswith("_G20")),
        None,
    )
    gp20_col = next(
        (
            c
            for c in df.columns
            if (("gamesPlayed" in c) or ("games_played" in c)) and c.endswith("_G20")
        ),
        None,
    )

    xga_pg_rec_goalie = pd.Series(np.nan, index=df.index)

    if ga10_col and gp10_col:
        ga10 = pd.to_numeric(df[ga10_col], errors="coerce")
        gp10 = pd.to_numeric(df[gp10_col], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            xga10_pg = np.where(gp10 > 0, ga10 / gp10, np.nan)
        xga_pg_rec_goalie = pd.Series(xga10_pg, index=df.index)

    if ga20_col and gp20_col:
        ga20 = pd.to_numeric(df[ga20_col], errors="coerce")
        gp20 = pd.to_numeric(df[gp20_col], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            xga20_pg = np.where(gp20 > 0, ga20 / gp20, np.nan)
        xga20_pg = pd.Series(xga20_pg, index=df.index)

        mask_existing = xga_pg_rec_goalie.notna()
        # Where we don't have L10, use L20
        xga_pg_rec_goalie[~mask_existing] = xga20_pg[~mask_existing]
        # Where we have both, blend
        xga_pg_rec_goalie[mask_existing] = (
            0.65 * xga_pg_rec_goalie[mask_existing] + 0.35 * xga20_pg[mask_existing]
        )

    df["xGA_pg_recency_goalie"] = xga_pg_rec_goalie

    # Derive team-level xGA_pg_recency from goalie recency if TEAM present
    if "TEAM" in df.columns:
        team_means = (
            df.groupby("TEAM")["xGA_pg_recency_goalie"].mean(numeric_only=True)
        )
        df["xGA_pg_recency"] = df["TEAM"].map(team_means)

    return df


# ---------------------------------------------------------
# Build Final Merged Pool
# ---------------------------------------------------------
def build_merged_player_pool() -> pd.DataFrame:
    print("\n========== MERGE PIPELINE START ==========\n")

    # Load RW
    df_rw = load_rotowire(RW_FILE)

    # ---- Load MP ----
    print("[STEP] Loading MoneyPuck L10/L20...")
    sk10 = load_skaters_l10()
    sk20 = load_skaters_l20()
    g10 = load_goalies_l10()
    g20 = load_goalies_l20()

    # ---- Normalize MP names ----
    for mp in [sk10, sk20, g10, g20]:
        if "name" in mp.columns:
            mp["PLAYER"] = mp["name"].apply(_normalize_mp_name).str.strip()
        else:
            mp["PLAYER"] = None

    # ---- Merge Steps ----
    print("[STEP] Merging RW + Skaters L10...")
    df = df_rw.merge(sk10, how="left", on="PLAYER", suffixes=("", "_L10"))

    print("[STEP] Merging RW + Skaters L20...")
    df = df.merge(sk20, how="left", on="PLAYER", suffixes=("", "_L20"))

    print("[STEP] Merging Goalie L10...")
    df = df.merge(g10, how="left", on="PLAYER", suffixes=("", "_G10"))

    print("[STEP] Merging Goalie L20...")
    df = df.merge(g20, how="left", on="PLAYER", suffixes=("", "_G20"))

    # ---- Build recency features from merged MP data ----
    print("[STEP] Deriving recency features (L10/L20 → *_recency)...")
    df = add_recency_features(df)

    print(f"[OK] Final merged size: {len(df)} players")
    print("\n========== MERGE PIPELINE COMPLETE ==========\n")

    return df


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("[RUN] Building merged player pool…")

    df = build_merged_player_pool()
    df.to_csv(MERGED_OUT, index=False)

    print(f"[DONE] Saved merged CSV → {MERGED_OUT}")


if __name__ == "__main__":
    main()
