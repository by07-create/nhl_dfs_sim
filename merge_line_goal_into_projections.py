# merge_line_goal_into_projections.py
"""
Merge line_goal_model.csv (lambda_total, p_goal_line)
into nhl_fd_projections.csv, matching by TEAM + LINE.

- Skaters on L1/L2/L3/L4 get non-zero lambda_total / p_goal_line
- Goalies / BN / players without a numeric LINE stay at 0
"""

import pandas as pd
import numpy as np
from pathlib import Path

APP_ROOT = Path.home() / "OneDrive" / "Documents" / "Bo - Python Apps" / "NHL Simulator"
PROJ_FILE = APP_ROOT / "nhl_fd_projections.csv"
LINE_FILE = APP_ROOT / "line_goal_model.csv"
OUT_FILE = PROJ_FILE  # overwrite in place


def build_team_line_keys_for_fd(df: pd.DataFrame) -> pd.DataFrame:
    """Add TEAM_KEY and LINE_KEY to nhl_fd_projections."""
    if "TEAM" not in df.columns:
        raise SystemExit("nhl_fd_projections.csv is missing TEAM column.")

    df["TEAM_KEY"] = df["TEAM"].astype(str).str.upper().str.strip()

    # Figure out which column represents the even-strength line number
    line_col = None
    for cand in ["LINE", "line_num", "Line", "line"]:
        if cand in df.columns:
            line_col = cand
            break

    if line_col is not None:
        line_numeric = pd.to_numeric(df[line_col], errors="coerce")
        # Build keys like L1, L2, L3, L4 for skaters only
        # SAFE HANDLING OF LINE_NUMERIC
        # ---------------------------
        line_numeric_clean = line_numeric.dropna()

        df["LINE_KEY"] = None
        df.loc[line_numeric.notna(), "LINE_KEY"] = (
            "L" + line_numeric.loc[line_numeric.notna()].astype(int).astype(str)
        )

    elif "env_key" in df.columns:
        # Fallback: derive from env_key like COL_L1 -> L1
        tmp = df["env_key"].astype(str).str.extract(r"_(L\d+)", expand=False)
        df["LINE_KEY"] = tmp.str.upper()
    else:
        # No line info at all - you'll just get zeros from the merge
        df["LINE_KEY"] = pd.NA

    # -------------------------------------
    #  PATCH: If LINE_KEY still missing, derive from env_key another way
    # -------------------------------------
    if df["LINE_KEY"].isna().all() and "env_key" in df.columns:
        tmp2 = df["env_key"].astype(str).str.extract(r"_L(\d+)", expand=False)
        df["LINE_KEY"] = tmp2.apply(lambda x: f"L{x}" if pd.notna(x) else None)

    # Explicitly ensure goalies do NOT get a line key
    pos_col = None
    for cand in ["POS", "Position", "base_pos"]:
        if cand in df.columns:
            pos_col = cand
            break
    if pos_col is not None:
        is_goalie = df[pos_col].astype(str).str.upper().eq("G")
        df.loc[is_goalie, "LINE_KEY"] = pd.NA

    return df


def build_team_line_keys_for_line_model(lm: pd.DataFrame) -> pd.DataFrame:
    """Make sure line_goal_model has TEAM_KEY and LINE_KEY."""
    if "TEAM_KEY" not in lm.columns:
        team_src = "TEAM" if "TEAM" in lm.columns else ("team" if "team" in lm.columns else None)
        if team_src is None:
            raise SystemExit("line_goal_model.csv missing TEAM/TEAM_KEY column.")
        lm["TEAM_KEY"] = lm[team_src].astype(str).str.upper().str.strip()

    if "LINE_KEY" not in lm.columns:
        if "line_tag" in lm.columns:
            lm["LINE_KEY"] = lm["line_tag"].astype(str).str.upper().str.strip()
        elif "env_key" in lm.columns:
            tmp = lm["env_key"].astype(str).str.extract(r"_(L\d+)", expand=False)
            lm["LINE_KEY"] = tmp.str.upper()
        else:
            raise SystemExit("line_goal_model.csv missing line_tag/LINE_KEY/env_key.")

    return lm


def main():
    if not PROJ_FILE.exists():
        raise SystemExit(f"Missing nhl_fd_projections.csv at {PROJ_FILE}")
    if not LINE_FILE.exists():
        raise SystemExit(f"Missing line_goal_model.csv at {LINE_FILE}")

    print(f" Loading FD projections -> {PROJ_FILE}")
    df = pd.read_csv(PROJ_FILE)

    print(f" Loading line goal model -> {LINE_FILE}")
    lm = pd.read_csv(LINE_FILE)

    # Prepare keys on both sides
    df = build_team_line_keys_for_fd(df)
    lm = build_team_line_keys_for_line_model(lm)

    # We only need the key + goal fields from the line model
    cols_needed = [c for c in ["TEAM_KEY", "LINE_KEY", "p_goal_line", "lambda_total"] if c in lm.columns]
    lm_small = lm[cols_needed].copy()

    print(" Merging line goal model into FD projections...")
    merged = df.merge(
        lm_small,
        on=["TEAM_KEY", "LINE_KEY"],
        how="left",
        suffixes=("", "_line"),
    )

    # Clean numeric fields; goalies / unmatched rows -> 0
    merged["p_goal_line"] = pd.to_numeric(merged.get("p_goal_line"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    merged["lambda_total"] = pd.to_numeric(merged.get("lambda_total"), errors="coerce").fillna(0.0)

    # Drop helper keys from the export
    merged.drop(columns=["TEAM_KEY", "LINE_KEY"], inplace=True, errors="ignore")

    print(f" Writing merged FD projections -> {OUT_FILE}")
    merged.to_csv(OUT_FILE, index=False)
    print(" Done - lambda_total and p_goal_line attached to nhl_fd_projections.csv")


if __name__ == "__main__":
    main()
