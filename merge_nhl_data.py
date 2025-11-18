# merge_nhl_data.py — Cloud-Safe RW + MoneyPuck Merge
# Uses MoneyPuck L10 + L20 (2025 season)
# Fully compatible with Render / Streamlit Cloud

import pandas as pd
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
# Robust MoneyPuck CSV downloader (Retry + Fallback)
# ---------------------------------------------------------
def _load_mp_csv(url: str, label: str) -> pd.DataFrame:
    """
    Robust MoneyPuck CSV fetcher:
    - Adds headers (MoneyPuck blocks requests without UA)
    - 3 retries
    - fallback to cached CSV if MoneyPuck is unavailable
    """

    print(f"[INFO] Fetching MoneyPuck {label} → {url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64)"
        ),
        "Accept": "text/csv,*/*;q=0.9",
    }

    # Try 3 times before falling back
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code == 200:
                text = r.text.strip()

                if len(text) < 30:
                    print(f"[WARN] Empty MoneyPuck CSV for {label}, retrying...")
                else:
                    df = pd.read_csv(StringIO(text))
                    df.to_csv(_cache_path(label), index=False)
                    print(f"[OK] Loaded {label}: {len(df)} rows")
                    return df
            else:
                print(f"[WARN] HTTP {r.status_code} for {label}, retrying...")

        except Exception as e:
            print(f"[WARN] Attempt {attempt+1} failed for {label}: {e}")

        time.sleep(2)

    # Fallback
    cache_file = _cache_path(label)
    if cache_file.exists():
        print(f"[FALLBACK] Using cached MoneyPuck {label}: {cache_file}")
        return pd.read_csv(cache_file)

    raise RuntimeError(
        f"[ERROR] MoneyPuck failed for {label} after retries and no cache available."
    )


# ---------------------------------------------------------
# MoneyPuck L10 + L20 endpoints (Skaters + Goalies)
# ---------------------------------------------------------

def load_skaters_l10() -> pd.DataFrame:
    url = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/skaters_10.csv"
    return _load_mp_csv(url, "skaters_L10")

def load_skaters_l20() -> pd.DataFrame:
    url = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/skaters_20.csv"
    return _load_mp_csv(url, "skaters_L20")

def load_goalies_l10() -> pd.DataFrame:
    url = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies_10.csv"
    return _load_mp_csv(url, "goalies_L10")

def load_goalies_l20() -> pd.DataFrame:
    url = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies_20.csv"
    return _load_mp_csv(url, "goalies_L20")


# ---------------------------------------------------------
# Load Rotowire
# ---------------------------------------------------------
def load_rotowire(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Rotowire file not found: {path}")

    print(f"[INFO] Loading Rotowire slate → {path}")
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    print(f"[OK] RW count: {len(df)}")
    return df


# ---------------------------------------------------------
# Build Full Merged Player Pool
# ---------------------------------------------------------
def build_merged_player_pool() -> pd.DataFrame:
    print("[STEP] Loading Rotowire…")
    df_rw = load_rotowire(RW_FILE)

    print("[STEP] Loading MoneyPuck L10…")
    df_skaters10 = load_skaters_l10()
    df_goalies10 = load_goalies_l10()

    print("[STEP] Loading MoneyPuck L20…")
    df_skaters20 = load_skaters_l20()
    df_goalies20 = load_goalies_l20()

    # Standardize names
    for mp in [df_skaters10, df_skaters20, df_goalies10, df_goalies20]:
        if "name" in mp.columns:
            mp["PLAYER"] = mp["name"].astype(str).str.strip()

    # Merge L10 skaters
    print("[STEP] Merge RW + MP Skaters L10")
    df = df_rw.merge(
        df_skaters10,
        how="left",
        on="PLAYER",
        suffixes=("", "_L10"),
    )

    # Merge L20 skaters
    print("[STEP] Merge RW + MP Skaters L20")
    df = df.merge(
        df_skaters20,
        how="left",
        on="PLAYER",
        suffixes=("", "_L20"),
    )

    # Merge goalies L10
    print("[STEP] Merge Goalie L10")
    df = df.merge(
        df_goalies10,
        how="left",
        on="PLAYER",
        suffixes=("", "_G_L10"),
    )

    # Merge goalies L20
    print("[STEP] Merge Goalie L20")
    df = df.merge(
        df_goalies20,
        how="left",
        on="PLAYER",
        suffixes=("", "_G_L20"),
    )

    print(f"[OK] Final merged player count: {len(df)}")
    return df


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("[RUN] Building merged player pool…")

    df = build_merged_player_pool()

    df.to_csv(MERGED_OUT, index=False)
    print(f"[DONE] Saved merged file → {MERGED_OUT}")


if __name__ == "__main__":
    main()
