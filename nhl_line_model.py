# nhl_line_model.py
#
# Recency-aware Poisson line goal model + chemistry score.
#
# Uses:
#   - nhl_player_projections.csv (RW + MoneyPuck + recency already blended)
#
# Outputs:
#   - line_goal_model.csv with:
#       TEAM, env_key, line_tag
#       lambda_total       (expected goals for the line)
#       p_goal_line        (prob line scores >= 1 goal)
#       chemistry_score    (assist-based synergy proxy)
#       TEAM_KEY, LINE_KEY (for merges)
#
# merge_line_goal_into_projections.py can keep working off this file.

import numpy as np
import pandas as pd
from pathlib import Path

APP_ROOT = Path.home() / "OneDrive" / "Documents" / "Bo - Python Apps" / "NHL Simulator"

INPUT_FILE = APP_ROOT / "nhl_player_projections.csv"
OUTPUT_FILE = APP_ROOT / "line_goal_model.csv"

# Config
D_WEIGHT = 0.40          # Defensemen offensive weight
DEF_CLAMP = (0.60, 1.40) # clamp for defensive multiplier base
ENV_CLAMP = (0.85, 1.25) # clamp for O/U environment
HOME_MULT = 1.05
AWAY_MULT = 0.95


def load_projections() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # Basic sanity columns
    for col in ["PLAYER", "TEAM", "OPP", "env_key"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in projections file.")

    # Normalize strings
    df["TEAM"] = df["TEAM"].astype(str).str.upper()
    df["OPP"] = df["OPP"].astype(str).str.upper()
    df["env_key"] = df["env_key"].astype(str)

    # Make sure position info is there
    if "base_pos" not in df.columns:
        # Fallback: try POS, otherwise treat everyone as skater
        if "POS" in df.columns:
            df["base_pos"] = df["POS"].astype(str).str.strip().str.upper().str[0]
        else:
            df["base_pos"] = "W"

    # Numeric fields we care about
    numeric_cols = [
        "xG_per60",
        "xG_per60_recency",
        "icetime",
        "games_played",
        "I_F_primaryAssists",
        "I_F_secondaryAssists",
        "team_offense_mult",
        "line_offense_mult",
        "xGA_pg_recency",
        "xGoalsAgainst_teamenv",
        "O/U",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # is_home is optional; keep as bool where possible
    if "is_home" in df.columns:
        df["is_home"] = df["is_home"].astype(bool)
    else:
        df["is_home"] = False

    return df


def compute_line_model(df: pd.DataFrame) -> pd.DataFrame:
    # Work with skaters only; ignore goalies here
    sk = df.copy()
    sk = sk[sk["env_key"].notna() & (sk["env_key"] != "")]

    if sk.empty:
        raise RuntimeError("No skaters with env_key found in projections.")

    # TOI per game (clamped)
    toi_pg = np.where(
        sk["games_played"] > 0,
        (sk["icetime"] / 60.0) / sk["games_played"],
        np.nan,
    )
    toi_pg = pd.Series(toi_pg, index=sk.index).fillna(16.0).clip(8.0, 24.0)
    sk["_toi_pg"] = toi_pg

    # Recency-aware xG per 60: prefer recency, fallback to season
    xg60_rec = pd.to_numeric(sk.get("xG_per60_recency", np.nan), errors="coerce")
    xg60_season = pd.to_numeric(sk.get("xG_per60", np.nan), errors="coerce")

    xg60 = np.where(
        np.isfinite(xg60_rec) & (xg60_rec > 0),
        xg60_rec,
        xg60_season,
    )
    xg60 = pd.Series(xg60, index=sk.index).fillna(0.0)

    # Player xG per game from xG/60 * TOI
    sk["_xg_pg"] = xg60 * (sk["_toi_pg"] / 60.0)

    # Assist-based chemistry components per 60 (if present)
    pri_ast = pd.to_numeric(sk.get("I_F_primaryAssists", 0.0), errors="coerce")
    sec_ast = pd.to_numeric(sk.get("I_F_secondaryAssists", 0.0), errors="coerce")
    gp = pd.to_numeric(sk.get("games_played", 0.0), errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        pri_ast60 = np.where(
            (gp > 0),
            pri_ast / gp / (sk["_toi_pg"] / 60.0).replace(0, np.nan),
            np.nan,
        )
        sec_ast60 = np.where(
            (gp > 0),
            sec_ast / gp / (sk["_toi_pg"] / 60.0).replace(0, np.nan),
            np.nan,
        )

    sk["_prim_ast60"] = np.nan_to_num(pri_ast60, nan=0.0)
    sk["_sec_ast60"] = np.nan_to_num(sec_ast60, nan=0.0)

    # Group by line (TEAM, env_key)
    groups = sk.groupby(["TEAM", "env_key"], sort=False)
    rows = []

    for (team, env_key), sub in groups:
        if "_" in env_key:
            _, line_tag = env_key.split("_", 1)
        else:
            line_tag = env_key

        opp = str(sub["OPP"].iloc[0]).upper()
        is_home = bool(sub["is_home"].iloc[0]) if "is_home" in sub.columns else False

        # -----------------------
        # 1) OFFENSE (FORWARDS + D)
        # -----------------------
        forwards = sub[sub["base_pos"].isin(["C", "W"])]
        defense = sub[sub["base_pos"].eq("D")]

        fwd_xg = forwards["_xg_pg"].sum()
        d_xg = defense["_xg_pg"].sum() * D_WEIGHT
        raw_xg_off = fwd_xg + d_xg

        if raw_xg_off <= 0:
            # No offensive contribution -> skip line
            continue

        team_off = sub["team_offense_mult"].replace(0, 1.0).mean()
        line_off = sub["line_offense_mult"].replace(0, 1.0).mean()

        lambda_off = raw_xg_off * team_off * line_off

        # -----------------------
        # 2) DEFENSIVE SUPPRESSION (RECENCY-FIRST)
        # -----------------------
        opp_rows = df[df["TEAM"] == opp]

        if not opp_rows.empty:
            xga_rec = pd.to_numeric(opp_rows.get("xGA_pg_recency", np.nan), errors="coerce")
            xga_season = pd.to_numeric(
                opp_rows.get("xGoalsAgainst_teamenv", np.nan),
                errors="coerce",
            )

            if np.isfinite(xga_rec).any():
                xga_team = xga_rec.replace(0, np.nan).mean()
            elif np.isfinite(xga_season).any():
                xga_team = xga_season.replace(0, np.nan).mean()
            else:
                xga_team = 2.8
        else:
            xga_team = 2.8  # fallback if no opp rows

        league_xga = 2.8
        base_def_mult = xga_team / league_xga
        base_def_mult = float(np.clip(base_def_mult, DEF_CLAMP[0], DEF_CLAMP[1]))

        # Slightly non-linear to accentuate strong D/weak D differences
        def_suppress = base_def_mult**2

        # -----------------------
        # 3) GAME ENVIRONMENT (O/U + HOME/AWAY)
        # -----------------------
        ou = pd.to_numeric(sub.get("O/U", np.nan), errors="coerce").mean()
        if np.isfinite(ou):
            env_mult = np.clip(ou / 6.0, ENV_CLAMP[0], ENV_CLAMP[1])
        else:
            env_mult = 1.0

        if is_home:
            env_mult *= HOME_MULT
        else:
            env_mult *= AWAY_MULT

        # -----------------------
        # 4) FINAL  AND GOAL PROB
        # -----------------------
        lambda_total = lambda_off * def_suppress * env_mult
        lambda_total = max(lambda_total, 0.0)

        p_goal_line = 1.0 - np.exp(-lambda_total)

        # -----------------------
        # 5) CHEMISTRY SCORE (ASSIST-BASED)
        # -----------------------
        # Use forwards only; assists per 60 as proxy for playmaking synergy.
        f_forwards = forwards.copy()
        if not f_forwards.empty:
            chem_raw = (f_forwards["_prim_ast60"] + 0.5 * f_forwards["_sec_ast60"]).mean()
        else:
            chem_raw = 0.0

        chem_score = chem_raw * team_off * line_off

        rows.append(
            {
                "TEAM": team,
                "env_key": env_key,
                "line_tag": line_tag,
                "lambda_total": lambda_total,
                "p_goal_line": p_goal_line,
                "chemistry_score": chem_score,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No line rows produced; check env_key / TEAM / OPP mappings.")

    out["TEAM_KEY"] = out["TEAM"].astype(str).str.upper()
    out["LINE_KEY"] = out["line_tag"].astype(str).str.upper()
    return out


def main():
    print(f" Loading player projections -> {INPUT_FILE}")
    df = load_projections()

    print(" Building recency-aware line goal model with chemistry")
    df_lines = compute_line_model(df)

    print(f" Writing line goal model -> {OUTPUT_FILE}")
    df_lines.to_csv(OUTPUT_FILE, index=False)

    print(" Done.")

if __name__ == "__main__":
    main()
