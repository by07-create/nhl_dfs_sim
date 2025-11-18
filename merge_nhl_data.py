# merge_nhl_data.py -- Cloud-Safe Version
# RW + MoneyPuck unified merge script for your NHL DFS pipeline
# This version works on Render, Streamlit Cloud, Linux, mobile, Windows, Mac

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from io import StringIO

# ---------------------------------------------------------
# CLOUD-SAFE DATA DIRECTORY
# ---------------------------------------------------------
DATA_DIR = Path(__file__).parent
RW_FILE = DATA_DIR / "rw-nhl-player-pool.xlsx"

# Output files (same folder, cloud-safe)
MERGED_OUT = DATA_DIR / "merged_nhl_player_pool.csv"


# ---------------------------------------------------------
# Robust MoneyPuck downloader (Option A)
# ---------------------------------------------------------
def _load_mp_csv(url: str, label: str) -> pd.DataFrame:
    """
    Robust MoneyPuck CSV fetcher.
    - Adds proper headers
    - Avoids empty CSV issues
    - Works on Render / Streamlit Cloud
    """
    print(f"[INFO] Fetching MoneyPuck {label} from: {url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/118.0 Safari/537.36"
        ),
        "Accept": "text/csv,*/*;q=0.9",
    }

    try:
        r = requests.get(url, headers=headers, timeout=30)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Connection error for {label}: {e}")

    if r.status_code != 200:
        raise RuntimeError(
            f"[ERROR] MoneyPuck HTTP {r.status_code} for {label}"
        )

    text = r.text.strip()
    if len(text) < 30:
        raise RuntimeError(
            f"[ERROR] MoneyPuck returned an EMPTY CSV for {label}. "
            f"Site may be blocking cloud requests."
        )

    try:
        df = pd.read_csv(StringIO(text))
    except Exception as e:
        raise RuntimeError(f"[ERROR] CSV parse failure for {label}: {e}")

    if df.empty:
        raise RuntimeError(
            f"[ERROR] {label} is empty after parsing. MP unavailable."
        )

    print(f"[OK] Loaded MoneyPuck {label}, rows: {len(df)}")
    return df


# ---------------------------------------------------------
# Rotowire Loader
# ---------------------------------------------------------
def load_rotowire(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"[ERROR] Rotowire file not found: {path}"
        )

    print(f"[INFO] Loading Rotowire slate: {path}")

    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    print(f"[OK] RW rows: {len(df)}")
    return df


# ---------------------------------------------------------
# MoneyPuck loader wrappers
# ---------------------------------------------------------
def load_skaters() -> pd.DataFrame:
    url = (
        "https://moneypuck.com/moneypuck/playerData/seasonSummary/2024/skaters.csv"
    )
    return _load_mp_csv(url, "skaters seasonSummary")


def load_goalies() -> pd.DataFrame:
    url = (
        "https://moneypuck.com/moneypuck/playerData/seasonSummary/2024/goalies.csv"
    )
    return _load_mp_csv(url, "goalies seasonSummary")


def load_onice() -> pd.DataFrame:
    url = (
        "https://moneypuck.com/moneypuck/playerData/onIce/2024/onIce_all.csv"
    )
    return _load_mp_csv(url, "on-ice data")


# ---------------------------------------------------------
# Build merged RW + MP dataset
# ---------------------------------------------------------
def build_merged_player_pool() -> pd.DataFrame:
    print("[STEP] Load RW data")
    df_rw = load_rotowire(RW_FILE)

    print("[STEP] Load MoneyPuck skaters")
    df_skaters = load_skaters()

    print("[STEP] Load MoneyPuck goalies")
    df_goalies = load_goalies()

    print("[STEP] Load MoneyPuck on-ice matchup data")
    df_onice = load_onice()

    # Standardize names
    if "name" in df_skaters.columns:
        df_skaters["PLAYER"] = df_skaters["name"].astype(str).str.strip()
    if "name" in df_goalies.columns:
        df_goalies["PLAYER"] = df_goalies["name"].astype(str).str.strip()

    # Merge RW with MP skaters
    print("[STEP] Merging RW with MoneyPuck skaters...")
    df = df_rw.merge(
        df_skaters,
        how="left",
        left_on="PLAYER",
        right_on="PLAYER",
    )

    # Attach goalie data
    print("[STEP] Merging RW with MoneyPuck goalies...")
    df = df.merge(
        df_goalies,
        how="left",
        left_on="PLAYER",
        right_on="PLAYER",
        suffixes=("", "_goalie"),
    )

    # Attach on-ice data
    if "PLAYER" in df_onice.columns:
        print("[STEP] Merging RW with MoneyPuck on-ice...")
        df = df.merge(
            df_onice,
            how="left",
            left_on="PLAYER",
            right_on="PLAYER",
            suffixes=("", "_onice"),
        )

    print(f"[OK] Final merged rows: {len(df)}")
    return df


# ---------------------------------------------------------
# Main runner
# ---------------------------------------------------------
def main():
    print("[RUN] Starting RW + MoneyPuck merge...")

    df = build_merged_player_pool()

    df.to_csv(MERGED_OUT, index=False)
    print(f"[DONE] Saved merged player pool → {MERGED_OUT}")


if __name__ == "__main__":
    main()