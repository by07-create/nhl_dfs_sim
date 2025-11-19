# nhl_export_for_fd.py (CLOUD-SAFE PATH PATCH — NO LOGIC CHANGES)

import pandas as pd
from pathlib import Path

# -------------------------------------
# Cloud-safe project root
# -------------------------------------
PROJECT_DIR = Path(__file__).parent.resolve()

# Input/output now live inside app directory
INPUT_FILE  = PROJECT_DIR / "nhl_player_projections.csv"
OUTPUT_FILE = PROJECT_DIR / "nhl_fd_projections.csv"


def main():
    print(f" Loading projections -> {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    # -------------------------------------
    # Pick the main FD projection:
    # 1. final_fpts (full blended model)
    # 2. prop_adj_fpts (Vegas-adjusted)
    # 3. FPTS (Rotowire fallback)
    # -------------------------------------
    if "final_fpts" in df.columns:
        df["PROJ"] = df["final_fpts"]
    elif "prop_adj_fpts" in df.columns:
        df["PROJ"] = df["prop_adj_fpts"]
    else:
        df["PROJ"] = df.get("FPTS", 0)

    df["PROJ"] = pd.to_numeric(df["PROJ"], errors="coerce").fillna(0)

    # -------------------------------------
    # Columns we include in the FD export
    # -------------------------------------
    keep_cols = [
        "PLAYER",
        "POS",
        "TEAM",
        "OPP",
        "SAL",
        "PROJ",
        "is_goalie",
        "FPTS",
        "fpts_sigma",
        "final_fpts",
        "prop_adj_fpts",
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

    # Keep only existing columns
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_out = df[keep_cols].copy()

    # -------------------------------------
    # Preserve line/env information
    # (required by merge_line_goal_into_projections.py)
    # -------------------------------------
    if "env_key" in df.columns:
        df_out["env_key"] = df["env_key"]

    if "line_num" in df.columns:
        df_out["line_num"] = df["line_num"]
    elif "LINE" in df.columns:
        df_out["line_num"] = df["LINE"]
    else:
        df_out["line_num"] = None

    # Sort best → worst projection
    df_out = df_out.sort_values("PROJ", ascending=False)

    print(f" Writing FanDuel export -> {OUTPUT_FILE}")
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(" Done.")


if __name__ == "__main__":
    main()
