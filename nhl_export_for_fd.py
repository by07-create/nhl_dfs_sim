# nhl_export_for_fd.py
#
# Cloud-safe FanDuel export. Ensures:
#   - final_fpts is exported
#   - fpts_sigma exported
#   - env_key + line_num exported
#   - base_pos exported
#   - recency + matchup + line strength fields exported
#   - No hard-coded Windows paths

import pandas as pd
from pathlib import Path

# -------------------------------------
# Project Paths (cloud-safe)
# -------------------------------------
APP_ROOT = Path(__file__).parent.resolve()

INPUT_FILE = APP_ROOT / "nhl_player_projections.csv"
OUTPUT_FILE = APP_ROOT / "nhl_fd_projections.csv"


def main():
    print(f" Loading projections -> {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    # -------------------------------------
    # PROJ column (used by FD builder)
    # Order:
    #   1. final_fpts
    #   2. prop_adj_fpts
    #   3. FPTS (raw Rotowire)
    # -------------------------------------
    if "final_fpts" in df.columns:
        df["PROJ"] = df["final_fpts"]
    elif "prop_adj_fpts" in df.columns:
        df["PROJ"] = df["prop_adj_fpts"]
    else:
        df["PROJ"] = df.get("FPTS", 0)

    df["PROJ"] = pd.to_numeric(df["PROJ"], errors="coerce").fillna(0.0)

    # -------------------------------------
    # Guarantee core identifiers
    # -------------------------------------
    # POS → base_pos fallback
    if "POS" not in df.columns and "base_pos" in df.columns:
        df["POS"] = df["base_pos"]

    if "base_pos" not in df.columns and "POS" in df.columns:
        df["base_pos"] = df["POS"].astype(str).str.upper().str[0]

    # -------------------------------------
    # FD Export Columns
    # -------------------------------------
    keep_cols = [
        "PLAYER",
        "POS",
        "base_pos",
        "TEAM",
        "OPP",
        "SAL",
        "PROJ",

        # Core projection components
        "final_fpts",
        "prop_adj_fpts",
        "FPTS",
        "fpts_sigma",

        # Matchup multipliers
        "matchup_mult",
        "line_strength_mult",
        "final_matchup_mult",
        "line_strength_norm",
        "line_matchup_strength",

        # Keep recency signals (useful for debugging + slate sim)
        "xG_per60_recency",
        "SOG_per60_recency",
        "xGA_pg_recency",
        "xGA_pg_recency_goalie",
    ]

    # Only keep columns that actually exist
    keep_cols = [c for c in keep_cols if c in df.columns]

    df_out = df[keep_cols].copy()

    # -------------------------------------
    # Preserve env_key + line_num (essential for line model merge)
    # -------------------------------------
    if "env_key" in df.columns:
        df_out["env_key"] = df["env_key"]

    # Find line number field
    if "line_num" in df.columns:
        df_out["line_num"] = df["line_num"]
    elif "LINE" in df.columns:
        df_out["line_num"] = df["LINE"]
    else:
        df_out["line_num"] = None

    # -------------------------------------
    # Sort FD file best → worst
    # -------------------------------------
    df_out = df_out.sort_values("PROJ", ascending=False)

    print(f" Writing FanDuel export -> {OUTPUT_FILE}")
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(" Done.")


if __name__ == "__main__":
    main()
