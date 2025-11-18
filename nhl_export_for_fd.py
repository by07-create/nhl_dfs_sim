# nhl_export_for_fd.py

import pandas as pd
from pathlib import Path

# -------------------------------------
# Project Paths
# -------------------------------------
PROJECT_DIR = Path(r"C:\Users\b-yan\OneDrive\Documents\Bo - Python Apps\NHL Simulator")

INPUT_FILE = PROJECT_DIR / "nhl_player_projections.csv"
OUTPUT_FILE = PROJECT_DIR / "nhl_fd_projections.csv"


def main():
    print(f" Loading projections -> {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    # -------------------------------------
    # Create a clean, single projection column for FD
    # Order of importance:
    #   1. final_fpts (full model)
    #   2. prop_adj_fpts (if props adjusted it)
    #   3. Rotowire FPTS fallback
    # -------------------------------------
    if "final_fpts" in df.columns:
        df["PROJ"] = df["final_fpts"]
    elif "prop_adj_fpts" in df.columns:
        df["PROJ"] = df["prop_adj_fpts"]
    else:
        df["PROJ"] = df.get("FPTS", 0)

    # Make sure PROJ is numeric
    df["PROJ"] = pd.to_numeric(df["PROJ"], errors="coerce").fillna(0)

    # -------------------------------------
    # Columns we want for FanDuel upload
    # -------------------------------------
    keep_cols = [
        "PLAYER",
        "POS",
        "TEAM",
        "OPP",
        "SAL",
        "PROJ",  # <<< main projection used by lineup builder
        "is_goalie",
        "FPTS", # raw Rotowire
        "fpts_sigma",       
        "final_fpts",        # model projection
        "prop_adj_fpts",     # vegas-adjusted projection
        "xg_per_game_model",
        "shots_per_game_model",
        "assists_per_game_model",
        "blocks_per_game_model",
        "pp_points_per_game_model",
        "goalie_win_prob",
    
    "matchup_mult",
    "line_strength_norm",
    "line_matchup_strength",
]

    # Keep only columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_out = df[keep_cols].copy()

    # -------------------------------------
    #  PATCH: Preserve line information so merge_line_goal works
    # -------------------------------------
    # env_key example: "CAR_L1"
    if "env_key" in df.columns:
        df_out["env_key"] = df["env_key"]

    # line_num example: 1,2,3,4
    if "line_num" in df.columns:
        df_out["line_num"] = df["line_num"]
    elif "LINE" in df.columns:
        df_out["line_num"] = df["LINE"]
    else:
        df_out["line_num"] = None


    # Sort best -> worst for convenience
    df_out = df_out.sort_values("PROJ", ascending=False)

    print(f" Writing FanDuel export -> {OUTPUT_FILE}")
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(" Done.")


if __name__ == "__main__":
    main()
