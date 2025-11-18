# merge_nhl_data.py — Cloud-Safe RW + MoneyPuck Merge (L10/L20)
# Uses MoneyPuck recency (L10 + L20) for skaters and goalies.
# 100% portable for Render, Streamlit Cloud, Replit, etc.

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
        "skaters_L10"
    )

def load_skaters_l20():
    return _load_mp_csv(
        "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/skaters_20.csv",
        "skaters_L20"
    )

def load_goalies_l10():
    return _load_mp_csv(
        "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies_10.csv",
        "goalies_L10"
    )

def load_goalies_l20():
    return _load_mp_csv(
        "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies_20.csv",
        "goalies_L20"
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
    g10  = load_goalies_l10()
    g20  = load_goalies_l20()

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
