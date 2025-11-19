# nhl_line_model.py (recency-enhanced boom model)
#
# Line-level scoring / boom model for NHL DFS with:
#   - Recency-aware xG (player + line)
#   - Recency-aware team defensive xGA suppression
#   - Goalie shot-stopping multiplier with recency tilt
#   - FULL Line-vs-Line weighting (TOI-weighted)
#   - PP1 strength
#   - Recency-weighted high-danger chance profile
#   - PK weakness multiplier (via HDCA, recency-tilted)
#   - High-danger defense allowed (HDCA) multiplier
#   - Forward-only vs Forward+Defense (FLD) splits
#   - Line salary + value score
#
# Output columns include (key ones):
#   TEAM, OPP, env_key, line_role
#   line_proj_total, line_sim_mean, line_boom_pct, line_score
#   fwd_line_proj_total, fwd_boom_pct, fwd_score
#   fld_line_proj_total, fld_boom_pct, fld_score
#   line_xg_pg, line_hdcf, goalie_mult, lvs_mult
#   line_salary_total, line_salary_mean, line_value_score
#   game_class, vegas_total

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project paths (CLOUD-SAFE)
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.resolve()

INPUT_FILE = PROJECT_DIR / "nhl_player_projections.csv"
OUTPUT_FILE = PROJECT_DIR / "nhl_line_boom_model.csv"
MATCHUPS_FILE = PROJECT_DIR / "5v5_matchups_summary.csv"


# ---------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------
def load_projections(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing nhl_player_projections.csv: {path}")

    df = pd.read_csv(path)

    # Basic sanity
    for c in ["PLAYER", "TEAM"]:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' in {path}.")

    # Position handling
    if "base_pos" in df.columns:
        pos_col = "base_pos"
    elif "POS" in df.columns:
        pos_col = "POS"
    else:
        raise ValueError("Expected base_pos or POS column in nhl_player_projections.csv.")

    df["base_pos"] = df[pos_col].astype(str).str.upper()

    # Goalie flag
    if "is_goalie" not in df.columns:
        df["is_goalie"] = df["base_pos"].eq("G").astype(int)

    # Line key
    if "env_key" not in df.columns:
        raise ValueError("env_key column missing – required for line-level modeling.")

    df["env_key"] = df["env_key"].astype(str)
    df["TEAM"] = df["TEAM"].astype(str).str.upper()

    if "OPP" in df.columns:
        df["OPP"] = df["OPP"].astype(str).str.upper()
    else:
        df["OPP"] = ""

    # Projections
    if "final_fpts" in df.columns:
        df["final_fpts"] = pd.to_numeric(df["final_fpts"], errors="coerce").fillna(0.0)
    else:
        proj = df.get("PROJ", df.get("FPTS", 0))
        df["final_fpts"] = pd.to_numeric(proj, errors="coerce").fillna(0.0)

    # Sigma (vol)
    if "fpts_sigma" in df.columns:
        df["fpts_sigma"] = pd.to_numeric(df["fpts_sigma"], errors="coerce").fillna(0.0)
    else:
        df["fpts_sigma"] = df["final_fpts"].clip(lower=1.0) * 0.8

    return df


# ---------------------------------------------------------------------
# Load Vegas O/U from Rotowire file in Downloads
# ---------------------------------------------------------------------
def load_rw_vegas_totals() -> Dict[str, float]:
    """
    Load Vegas O/U directly from Rotowire NHL player pool file in Downloads.
    On cloud, this will usually just log "not found" and return {}.
    """
    rw_path = Path.home() / "Downloads" / "rw-nhl-player-pool.xlsx"
    if not rw_path.exists():
        print(f"RW file not found: {rw_path}")
        return {}

    try:
        df_rw = pd.read_excel(rw_path)
    except Exception as e:
        print(f"Failed loading RW file: {e}")
        return {}

    vegas_cols: List[str] = []
    for c in df_rw.columns:
        c_clean = str(c).lower().replace(" ", "").replace("/", "")
        if any(key in c_clean for key in ["ou", "overunder", "total", "vegas", "uline"]):
            vegas_cols.append(c)

    if not vegas_cols:
        print("No O/U column found in RW file.")
        return {}

    col = vegas_cols[0]
    print(f"Vegas O/U column detected: '{col}'")

    df_rw["_vegas"] = pd.to_numeric(df_rw[col], errors="coerce")

    out: Dict[str, float] = {}
    for team, sub in df_rw.groupby("TEAM"):
        vals = sub["_vegas"].dropna()
        if len(vals):
            out[str(team).upper()] = float(vals.iloc[0])

    print(f"Loaded {len(out)} Vegas totals.")
    return out


# ---------------------------------------------------------------------
# Load 5v5 line-vs-line matchup summary
# ---------------------------------------------------------------------
def load_matchups_summary() -> Optional[pd.DataFrame]:
    """Load 5v5 line-vs-line matchup summary if present."""
    if not MATCHUPS_FILE.exists():
        print(f"No 5v5 matchup summary found at {MATCHUPS_FILE}")
        return None

    df = pd.read_csv(MATCHUPS_FILE)

    # Normalize columns
    cols_lower = {c.lower(): c for c in df.columns}
    needed = ["team", "line", "opp_team", "opp_line", "toi"]
    rename_map = {}
    for need in needed:
        if need not in df.columns and need in cols_lower:
            rename_map[cols_lower[need]] = need
    if rename_map:
        df = df.rename(columns=rename_map)

    for c in needed:
        if c not in df.columns:
            print(f"5v5 summary missing '{c}' – line-vs-line weighting disabled.")
            return None

    df["team"] = df["team"].astype(str).str.upper()
    df["opp_team"] = df["opp_team"].astype(str).str.upper()
    df["line"] = df["line"].astype(str).str.upper()
    df["opp_line"] = df["opp_line"].astype(str).str.upper()
    df["toi"] = pd.to_numeric(df["toi"], errors="coerce").fillna(0.0)

    return df


# ---------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------
def simulate_line(means: np.ndarray, sigmas: np.ndarray, n_sims: int = 100000) -> Dict[str, float]:
    if n_sims <= 0 or len(means) == 0:
        return {
            "line_sim_mean": 0.0,
            "line_sim_std": 0.0,
            "line_sim_p75": 0.0,
            "line_sim_p90": 0.0,
            "line_sim_p95": 0.0,
            "line_boom_pct": 0.0,
        }

    means = means.astype(float)
    sigmas = sigmas.astype(float)
    scores = np.zeros(n_sims, dtype=float)

    for mu, sigma in zip(means, sigmas):
        draws = np.random.normal(mu, max(sigma, 0.1), n_sims)
        scores += np.clip(draws, 0.0, None)

    sim_mean = float(scores.mean())
    sim_std = float(scores.std(ddof=0))
    p75 = float(np.percentile(scores, 75))
    p90 = float(np.percentile(scores, 90))
    p95 = float(np.percentile(scores, 95))

    boom_threshold = 1.10 * float(means.sum())
    boom_pct = float((scores >= boom_threshold).mean())

    return {
        "line_sim_mean": sim_mean,
        "line_sim_std": sim_std,
        "line_sim_p75": p75,
        "line_sim_p90": p90,
        "line_sim_p95": p95,
        "line_boom_pct": boom_pct,
    }


# ---------------------------------------------------------------------
# Line-level summariser
# ---------------------------------------------------------------------
def build_line_model(
    df: pd.DataFrame,
    df_matchups: Optional[pd.DataFrame] = None,
    n_sims: int = 10000,
) -> pd.DataFrame:
    # Keep only skaters with a line env_key
    mask_skater = (df.get("is_goalie", 0) == 0) & df["env_key"].notna() & df["env_key"].ne("")
    skaters = df.loc[mask_skater].copy()

    if skaters.empty:
        raise RuntimeError("No skaters with env_key found.")

    # -------------------------
    # RECENCY-FIRST xG per 60
    # -------------------------
    xg60_rec = pd.to_numeric(skaters.get("xG_per60_recency", np.nan), errors="coerce")
    xg60_season = pd.to_numeric(skaters.get("xG_per60", np.nan), errors="coerce")

    # Prefer recency if it exists and is positive; otherwise fall back to season xG_per60
    xg60_arr = np.where(
        np.isfinite(xg60_rec) & (xg60_rec > 0),
        xg60_rec,
        xg60_season,
    )
    xg60 = pd.Series(xg60_arr, index=skaters.index).fillna(0.0)

    games_played = pd.to_numeric(skaters.get("games_played", np.nan), errors="coerce").fillna(0.0)
    icetime_sec = pd.to_numeric(skaters.get("icetime", np.nan), errors="coerce").fillna(0.0)
    matchup_mult = pd.to_numeric(skaters.get("matchup_mult", 1.0), errors="coerce").fillna(1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        toi_min_pg = np.where(
            games_played > 0,
            (icetime_sec / 60.0) / games_played,
            np.nan,
        )
    toi_min_pg = pd.Series(toi_min_pg, index=skaters.index)
    toi_min_pg = toi_min_pg.clip(lower=8.0, upper=24.0).fillna(16.0)

    # Player-level xG per game (recency-weighted)
    player_xg_pg = xg60 * (toi_min_pg / 60.0)

    # -------------------------
    # Recency multiplier for offensive intensity (used for assists & HD)
    # -------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        off_rec_mult = np.where(
            xg60_season > 0,
            xg60 / xg60_season,
            1.0,
        )
    off_rec_mult = np.clip(off_rec_mult, 0.6, 1.4)
    off_rec_mult = pd.Series(off_rec_mult, index=skaters.index).fillna(1.0)
    skaters["_off_rec_mult"] = off_rec_mult

    # -------------------------
    # Assist per-60 (season-derived, recency-scaled by off_rec_mult)
    # -------------------------
    if "primary_ast_per60" in skaters.columns:
        prim_ast60 = pd.to_numeric(skaters["primary_ast_per60"], errors="coerce").fillna(0.0)
    else:
        pri_tot = pd.to_numeric(skaters.get("I_F_primaryAssists", 0.0), errors="coerce").fillna(0.0)
        gp = games_played.replace(0, np.nan)
        mins_pg = toi_min_pg.replace(0, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            prim_ast60 = (pri_tot / gp) / (mins_pg / 60.0)
        prim_ast60 = pd.Series(np.nan_to_num(prim_ast60, nan=0.0), index=skaters.index)

    if "secondary_ast_per60" in skaters.columns:
        sec_ast60 = pd.to_numeric(skaters["secondary_ast_per60"], errors="coerce").fillna(0.0)
    else:
        sec_tot = pd.to_numeric(skaters.get("I_F_secondaryAssists", 0.0), errors="coerce").fillna(0.0)
        gp = games_played.replace(0, np.nan)
        mins_pg = toi_min_pg.replace(0, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            sec_ast60 = (sec_tot / gp) / (mins_pg / 60.0)
        sec_ast60 = pd.Series(np.nan_to_num(sec_ast60, nan=0.0), index=skaters.index)

    # Recency scale assists by offensive recency multiplier
    prim_ast60 = prim_ast60 * off_rec_mult
    sec_ast60 = sec_ast60 * off_rec_mult

    team_off = pd.to_numeric(skaters.get("team_offense_mult", 1.0), errors="coerce").fillna(1.0)
    line_off = pd.to_numeric(skaters.get("line_offense_mult", 1.0), errors="coerce").fillna(1.0)

    skaters["_player_xg_pg"] = player_xg_pg
    skaters["_prim_ast60"] = prim_ast60
    skaters["_sec_ast60"] = sec_ast60
    skaters["_team_off"] = team_off
    skaters["_line_off"] = line_off
    skaters["_matchup_mult"] = matchup_mult

    # PP units
    if "pp_unit" in skaters.columns:
        skaters["pp_unit"] = skaters["pp_unit"]
    elif "PP LINE" in skaters.columns:
        skaters["pp_unit"] = skaters["PP LINE"]
    else:
        skaters["pp_unit"] = np.nan

    # Your existing line_strength + matchup_strength columns
    line_strength_norm = pd.to_numeric(
        skaters.get("line_strength_norm", np.nan), errors="coerce"
    ).fillna(0.0)
    line_matchup_strength_orig = pd.to_numeric(
        skaters.get("line_matchup_strength", np.nan), errors="coerce"
    ).fillna(0.0)

    skaters["_line_strength_norm"] = line_strength_norm
    skaters["_line_matchup_strength"] = line_matchup_strength_orig

    # Precompute a "defensive rating" for each line from your strength metric
    # This will be used for opponent matchup strength.
    line_def_table = (
        skaters.groupby(["TEAM", "env_key"])["_line_strength_norm"]
        .mean()
        .reset_index()
    )

    def_rating: Dict[tuple, float] = {}
    for _, row in line_def_table.iterrows():
        team = str(row["TEAM"])
        env = str(row["env_key"])
        if "_" not in env:
            continue
        _, role = env.split("_", 1)
        role = role.upper()
        if not (role.startswith("L") and role[1:].isdigit()):
            continue
        line_code = f"FL{role[1:]}"  # L1 -> FL1 etc.
        def_rating[(team, line_code)] = float(row["_line_strength_norm"])

    def_global_avg = float(line_def_table["_line_strength_norm"].mean()) if len(line_def_table) else 0.0

    # ------------------------------------------------------------------
    # League-average xGA for opponent defense (RECENCY-FIRST)
    # ------------------------------------------------------------------
    xga_team_all_rec = pd.to_numeric(df.get("xGA_pg_recency", np.nan), errors="coerce")
    xga_team_all_season = pd.to_numeric(df.get("xGoalsAgainst_teamenv", np.nan), errors="coerce")

    xga_team_all = np.where(
        np.isfinite(xga_team_all_rec) & (xga_team_all_rec > 0),
        xga_team_all_rec,
        xga_team_all_season,
    )
    xga_team_all_series = pd.Series(xga_team_all, index=df.index)
    league_xga = float(xga_team_all_series.replace(0, np.nan).mean())
    if not np.isfinite(league_xga) or league_xga <= 0:
        league_xga = 2.8  # reasonable NHL baseline

    # ------------------------------------------------------------------
    # League-average high-danger shots allowed, recency-tilted
    # ------------------------------------------------------------------
    if "highDangerShotsAgainst" in df.columns:
        hd_allowed_all = pd.to_numeric(df["highDangerShotsAgainst"], errors="coerce").fillna(0)
    else:
        hd_allowed_all = pd.Series([0.0] * len(df), index=df.index)

    if "shotAttemptsAgainst" in df.columns:
        shots_against_all = pd.to_numeric(df["shotAttemptsAgainst"], errors="coerce").replace(0, np.nan).fillna(1)
    else:
        shots_against_all = pd.Series([1.0] * len(df), index=df.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        hd_rate_all = (hd_allowed_all / shots_against_all).replace([np.inf, -np.inf], np.nan)

    # Use xGA recency vs season as a tilt on HD rate if possible
    with np.errstate(divide="ignore", invalid="ignore"):
        xga_ratio = np.where(
            xga_team_all_season > 0,
            xga_team_all_rec / xga_team_all_season,
            1.0,
        )
    xga_ratio = np.clip(xga_ratio, 0.7, 1.3)
    xga_ratio_series = pd.Series(xga_ratio, index=df.index).fillna(1.0)

    hd_rate_all_rec = hd_rate_all * xga_ratio_series
    league_hd_rate = float(hd_rate_all_rec.replace([np.inf, -np.inf], np.nan).mean())
    if not np.isfinite(league_hd_rate) or league_hd_rate <= 0:
        league_hd_rate = 0.20

    df["_hd_rate_rec"] = hd_rate_all_rec.fillna(league_hd_rate)

    # Group by line
    group_cols = ["TEAM", "OPP", "env_key"]
    groups = skaters.groupby(group_cols, sort=False)

    records: List[Dict[str, Any]] = []

    for (team, opp, env_key), sub in groups:
        if not isinstance(env_key, str) or env_key.strip() == "":
            continue

        sub = sub.copy()
        base_pos = sub["base_pos"].astype(str).str.upper()

        n_players = len(sub)
        n_forwards = int((base_pos.isin(["C", "W"])).sum())
        n_defense = int(base_pos.eq("D").sum())

        if "_" in env_key:
            _, role = env_key.split("_", 1)
        else:
            role = ""

        means = sub["final_fpts"].to_numpy(float)
        sigmas = sub["fpts_sigma"].to_numpy(float)

        line_proj_total = float(means.sum())
        line_proj_mean = float(means.mean()) if len(means) else 0.0

        line_xg_pg_raw = float(sub["_player_xg_pg"].sum())
        matchup_mean = float(sub["_matchup_mult"].mean()) if len(sub) else 1.0
        line_xg_pg = line_xg_pg_raw * matchup_mean

        chem_raw = (sub["_prim_ast60"] + 0.5 * sub["_sec_ast60"]).mean()
        team_off_mean = float(sub["_team_off"].mean()) if len(sub) else 1.0
        line_off_mean = float(sub["_line_off"].mean()) if len(sub) else 1.0
        chemistry_score = float(chem_raw * team_off_mean * line_off_mean)

        # -------------------------------
        # Recency-weighted High-Danger profile (full line)
        # -------------------------------
        off_mult_sub = pd.to_numeric(sub.get("_off_rec_mult", 1.0), errors="coerce").fillna(1.0)

        hd_low = pd.to_numeric(sub.get("I_F_lowDangerxGoals", 0), errors="coerce").fillna(0)
        hd_med = pd.to_numeric(sub.get("I_F_mediumDangerxGoals", 0), errors="coerce").fillna(0)
        hd_high = pd.to_numeric(sub.get("I_F_highDangerxGoals", 0), errors="coerce").fillna(0)

        # Scale danger events by offensive recency multiplier
        hd_low_adj = hd_low * off_mult_sub
        hd_med_adj = hd_med * off_mult_sub
        hd_high_adj = hd_high * off_mult_sub

        hd_total_adj = float((hd_low_adj + hd_med_adj + hd_high_adj).sum())
        hd_high_total_adj = float(hd_high_adj.sum())
        line_hdcf = (hd_high_total_adj / hd_total_adj) if hd_total_adj > 0 else 0.0

        # PP1 info
        pp_vals = pd.to_numeric(sub["pp_unit"], errors="coerce")
        pp1_count = int((pp_vals == 1).sum())
        has_pp1_core = pp1_count >= 2

        # -------------------------------
        # Opponent defense + goalie (recency-tilted)
        # -------------------------------
        # 1) Opponent team defensive environment (team xGA, RECENCY-FIRST)
        opp_rows_all = df[df["TEAM"] == opp]

        if len(opp_rows_all):
            opp_xga_rec_series = pd.to_numeric(
                opp_rows_all.get("xGA_pg_recency", np.nan),
                errors="coerce",
            )
            opp_xga_season_series = pd.to_numeric(
                opp_rows_all.get("xGoalsAgainst_teamenv", np.nan),
                errors="coerce",
            )
            opp_xga_series = np.where(
                np.isfinite(opp_xga_rec_series) & (opp_xga_rec_series > 0),
                opp_xga_rec_series,
                opp_xga_season_series,
            )
            opp_xga = float(pd.Series(opp_xga_series).replace(0, np.nan).mean())
        else:
            opp_xga = np.nan

        if np.isfinite(opp_xga) and league_xga > 0:
            # >1 = softer D, <1 = tougher D
            opp_def_mult = float(np.clip(opp_xga / league_xga, 0.8, 1.25))
        else:
            opp_def_mult = 1.0

        # 2) Opponent goalie shot-stopping (MoneyPuck xGoals vs goals per game, with recency tilt if available)
        opp_goalie_rows = df[(df["TEAM"] == opp) & (df["base_pos"] == "G")]

        goalie_mult_raw = 1.0
        if len(opp_goalie_rows):
            g_xg = pd.to_numeric(opp_goalie_rows.get("xGoals", np.nan), errors="coerce")
            g_goals = pd.to_numeric(opp_goalie_rows.get("goals", np.nan), errors="coerce")
            g_gp = pd.to_numeric(
                opp_goalie_rows.get("games_played_goalie", np.nan),
                errors="coerce",
            )

            g_xg = float(g_xg.replace(0, np.nan).mean())
            g_goals = float(g_goals.replace(0, np.nan).mean())
            g_gp = float(g_gp.replace(0, np.nan).mean())

            if np.isfinite(g_xg) and np.isfinite(g_goals) and np.isfinite(g_gp) and g_gp > 0 and g_xg > 0:
                g_xg_pg = g_xg / g_gp
                g_ga_pg = g_goals / g_gp
                # ratio > 1 means leaking more than expected (good for your skaters)
                ratio = g_ga_pg / g_xg_pg
                goalie_mult_raw = float(np.clip(ratio, 0.8, 1.25))
            else:
                goalie_mult_raw = 1.0

            # Recency tilt using goalie xGA_pg_recency_goalie if present
            if "xGA_pg_recency_goalie" in opp_goalie_rows.columns:
                g_xga_rec = pd.to_numeric(
                    opp_goalie_rows["xGA_pg_recency_goalie"],
                    errors="coerce",
                )
            else:
                g_xga_rec = pd.Series([np.nan] * len(opp_goalie_rows), index=opp_goalie_rows.index)
            g_xga_rec_mean = float(g_xga_rec.replace(0, np.nan).mean())
            if np.isfinite(g_xga_rec_mean) and league_xga > 0:
                rec_factor = float(np.clip(g_xga_rec_mean / league_xga, 0.8, 1.25))
                goalie_mult_raw *= rec_factor

        # --- HIGH-DANGER DEFENSE ALLOWED (HDCA) + PK WEAKNESS (RECENCY-TILTED) ---
        if len(opp_rows_all):
            opp_hd_rate_series = pd.to_numeric(
                opp_rows_all.get("_hd_rate_rec", np.nan),
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)
            opp_hd_rate = float(opp_hd_rate_series.mean())
        else:
            opp_hd_rate = league_hd_rate

        if not np.isfinite(opp_hd_rate) or opp_hd_rate <= 0:
            opp_hd_rate = league_hd_rate

        hdca_mult = float(np.clip(opp_hd_rate / league_hd_rate, 0.85, 1.30))
        pk_mult = float(np.clip(opp_hd_rate / league_hd_rate, 0.85, 1.35)) if pp1_count >= 2 else 1.0

        # 3) Combine team defense + goalie + HDCA + PK into one multiplier
        total_goalie_mult = opp_def_mult * goalie_mult_raw * hdca_mult * pk_mult
        total_goalie_mult = float(np.clip(total_goalie_mult, 0.75, 1.35))

        # -------------------------------
        # Line strength + matchup
        # -------------------------------
        ls_norm_mean = float(sub["_line_strength_norm"].mean())

        # Default matchup strength (if we can't compute from 5v5)
        lms_mean = float(sub["_line_matchup_strength"].mean())

        matchup_softness = 0.0
        lvs_mult = 1.0
        line_matchup_strength_mean = lms_mean

        # NEW: use 5v5 matchups to compute opponent defensive strength
        if (
            df_matchups is not None
            and isinstance(role, str)
            and role.upper().startswith("L")
            and role[1:].isdigit()
        ):
            line_code = f"FL{role[1:]}"  # our forward line code

            m = df_matchups[
                (df_matchups["team"] == team)
                & (df_matchups["opp_team"] == opp)
                & (df_matchups["line"] == line_code)
            ]

            if not m.empty and def_global_avg > 0:
                total_toi = float(m["toi"].sum())
                if total_toi > 0:
                    m = m.copy()
                    m["pct"] = m["toi"] / total_toi

                    def_vals = []
                    for _, row_m in m.iterrows():
                        opp_line = str(row_m["opp_line"])
                        pct = float(row_m["pct"])
                        def_val = def_rating.get((opp, opp_line), def_global_avg)
                        def_vals.append(pct * def_val)

                    line_matchup_strength_mean = float(sum(def_vals))

                    # softness: relative to league average defensive strength
                    matchup_softness = (
                        (line_matchup_strength_mean - def_global_avg) / def_global_avg
                    )

                    # Convert softness into small multiplier: easier matchups -> >1, tougher -> <1
                    # softness > 0 means tough (above avg def), softness < 0 means soft
                    lvs_mult = 1.0 - 0.5 * matchup_softness
                    lvs_mult = float(np.clip(lvs_mult, 0.9, 1.1))

        # -------------------------------
        # Forward-only vs Forward+Defense splits
        # -------------------------------
        # Forward-only subset
        sub_fwd = sub[base_pos.isin(["C", "W"])].copy()
        if len(sub_fwd):
            means_fwd = sub_fwd["final_fpts"].to_numpy(float)
            sigmas_fwd = sub_fwd["fpts_sigma"].to_numpy(float)
            fwd_proj_total = float(means_fwd.sum())
            fwd_xg_pg = float(sub_fwd["_player_xg_pg"].sum()) * matchup_mean

            off_mult_fwd = pd.to_numeric(sub_fwd.get("_off_rec_mult", 1.0), errors="coerce").fillna(1.0)
            fwd_hd_low = pd.to_numeric(sub_fwd.get("I_F_lowDangerxGoals", 0), errors="coerce").fillna(0)
            fwd_hd_med = pd.to_numeric(sub_fwd.get("I_F_mediumDangerxGoals", 0), errors="coerce").fillna(0)
            fwd_hd_high = pd.to_numeric(sub_fwd.get("I_F_highDangerxGoals", 0), errors="coerce").fillna(0)

            fwd_hd_low_adj = fwd_hd_low * off_mult_fwd
            fwd_hd_med_adj = fwd_hd_med * off_mult_fwd
            fwd_hd_high_adj = fwd_hd_high * off_mult_fwd

            fwd_hd_total = float((fwd_hd_low_adj + fwd_hd_med_adj + fwd_hd_high_adj).sum())
            fwd_hd_high_total = float(fwd_hd_high_adj.sum())
            fwd_line_hdcf = (fwd_hd_high_total / fwd_hd_total) if fwd_hd_total > 0 else 0.0
        else:
            means_fwd = np.array([], dtype=float)
            sigmas_fwd = np.array([], dtype=float)
            fwd_proj_total = 0.0
            fwd_xg_pg = 0.0
            fwd_line_hdcf = 0.0

        # Forward+Defense subset (FLD)
        sub_fld = sub[base_pos.isin(["C", "W", "D"])].copy()
        if len(sub_fld):
            means_fld = sub_fld["final_fpts"].to_numpy(float)
            sigmas_fld = sub_fld["fpts_sigma"].to_numpy(float)
            fld_proj_total = float(means_fld.sum())
            fld_xg_pg = float(sub_fld["_player_xg_pg"].sum()) * matchup_mean

            off_mult_fld = pd.to_numeric(sub_fld.get("_off_rec_mult", 1.0), errors="coerce").fillna(1.0)
            fld_hd_low = pd.to_numeric(sub_fld.get("I_F_lowDangerxGoals", 0), errors="coerce").fillna(0)
            fld_hd_med = pd.to_numeric(sub_fld.get("I_F_mediumDangerxGoals", 0), errors="coerce").fillna(0)
            fld_hd_high = pd.to_numeric(sub_fld.get("I_F_highDangerxGoals", 0), errors="coerce").fillna(0)

            fld_hd_low_adj = fld_hd_low * off_mult_fld
            fld_hd_med_adj = fld_hd_med * off_mult_fld
            fld_hd_high_adj = fld_hd_high * off_mult_fld

            fld_hd_total = float((fld_hd_low_adj + fld_hd_med_adj + fld_hd_high_adj).sum())
            fld_hd_high_total = float(fld_hd_high_adj.sum())
            fld_line_hdcf = (fld_hd_high_total / fld_hd_total) if fld_hd_total > 0 else 0.0
        else:
            means_fld = np.array([], dtype=float)
            sigmas_fld = np.array([], dtype=float)
            fld_proj_total = 0.0
            fld_xg_pg = 0.0
            fld_line_hdcf = 0.0

        # ----- Run MC sim for full line -----
        sim_stats = simulate_line(means, sigmas, n_sims=n_sims)

        # Forward-only MC
        sim_fwd = simulate_line(means_fwd, sigmas_fwd, n_sims=n_sims)
        # FLD MC
        sim_fld = simulate_line(means_fld, sigmas_fld, n_sims=n_sims)

        rec: Dict[str, Any] = {
            "TEAM": team,
            "OPP": opp,
            "env_key": env_key,
            "line_role": role,
            "n_players": n_players,
            "n_forwards": n_forwards,
            "n_defense": n_defense,
            "line_proj_total": line_proj_total,
            "line_proj_mean": line_proj_mean,
            "line_xg_pg": line_xg_pg,
            "matchup_mult_mean": matchup_mean,
            "line_strength_norm_mean": ls_norm_mean,
            "line_matchup_strength_mean": line_matchup_strength_mean,
            "chemistry_score": chemistry_score,
            "pp1_count": pp1_count,
            "has_pp1_core": has_pp1_core,
            "line_hdcf": line_hdcf,
            "goalie_mult": total_goalie_mult,
            "matchup_softness": matchup_softness,
            "lvs_mult": lvs_mult,
            # Forward-only splits
            "fwd_xg_pg": fwd_xg_pg,
            "fwd_line_hdcf": fwd_line_hdcf,
            "fwd_line_proj_total": fwd_proj_total,
            "fwd_boom_pct": sim_fwd["line_boom_pct"],
            "fwd_score": sim_fwd["line_sim_mean"] + 15.0 * sim_fwd["line_boom_pct"],
            # Forward+Defense (FLD) splits
            "fld_xg_pg": fld_xg_pg,
            "fld_line_hdcf": fld_line_hdcf,
            "fld_line_proj_total": fld_proj_total,
            "fld_boom_pct": sim_fld["line_boom_pct"],
            "fld_score": sim_fld["line_sim_mean"] + 15.0 * sim_fld["line_boom_pct"],
            **sim_stats,
        }

        # Composite line_score: boom-heavy, xG-aware, PP1 + danger + matchup + goalie
        base_score = (
            rec["line_sim_mean"]
            + 15.0 * rec["line_boom_pct"]
            + 0.5 * rec["line_xg_pg"]
            + 0.10 * rec["line_strength_norm_mean"]
            + 0.20 * rec["line_hdcf"]
            + 0.80 * rec["pp1_count"]
        )
        rec["line_score"] = base_score * rec["goalie_mult"] * rec["lvs_mult"]

        records.append(rec)

    df_lines = pd.DataFrame(records)
    df_lines = df_lines.sort_values("line_score", ascending=False).reset_index(drop=True)

    # -------------------------------------------------------------
    # DEFENSE STACK SCORING (which D pairs best with each forward line)
    # -------------------------------------------------------------

    # Split forwards and defense units
    df_fwd = df_lines[df_lines["line_role"].str.startswith('L')].copy()
    df_def = df_lines[df_lines["line_role"].str.startswith('P')].copy()

    # --------------------------
    # Compute DEF stack score
    # --------------------------
    df_def["best_def_for_line"] = df_def["env_key"]
    df_def["best_def_score"] = (
        df_def["fld_score"]
        + 8 * df_def["fld_line_hdcf"]
        + 10 * df_def["pp1_count"]
        + 5 * df_def["matchup_mult_mean"]
        + 6 * df_def["goalie_mult"]
        + 2 * df_def["lvs_mult"]
    )
    df_def["best_stack_score"] = df_def["best_def_score"]

    # --------------------------
    # Assign best DEF to FWD lines
    # --------------------------
    def_by_team = {
        team: sub.sort_values("best_def_score", ascending=False)
        for team, sub in df_def.groupby("TEAM")
    }

    best_def_list = []
    best_def_score_list = []

    for _, row in df_fwd.iterrows():
        team = row["TEAM"]
        if team not in def_by_team:
            best_def_list.append(None)
            best_def_score_list.append(0)
            continue
        best_def_row = def_by_team[team].iloc[0]
        best_def_list.append(best_def_row["env_key"])
        best_def_score_list.append(best_def_row["best_def_score"])

    df_fwd["best_def_for_line"] = best_def_list
    df_fwd["best_def_score"] = best_def_score_list

    # --------------------------
    # Merge FWD + DEF stack results back safely
    # --------------------------
    df_lines = df_lines.merge(
        pd.concat([df_fwd, df_def])[["TEAM", "env_key", "best_def_for_line", "best_def_score"]],
        on=["TEAM", "env_key"],
        how="left"
    )

    # Stack score = forward score + best def score (or def-only score)
    df_lines["best_stack_score"] = (
        df_lines["fwd_score"].fillna(0)
        + df_lines["best_def_score"].fillna(0)
    )

    # ----------------------------------------
    # Line salary + value metrics
    # ----------------------------------------
    try:
        if "SAL" in df.columns:
            skaters_all = df[
                (df.get("is_goalie", 0) == 0)
                & df["env_key"].notna()
                & df["env_key"].ne("")
            ]
            skaters_all = skaters_all.copy()
            skaters_all["SAL"] = pd.to_numeric(skaters_all["SAL"], errors="coerce").fillna(0.0)
            salary_grp = (
                skaters_all
                .groupby(["TEAM", "OPP", "env_key"], as_index=False)["SAL"]
                .agg(line_salary_total="sum")
            )
            df_lines = df_lines.merge(salary_grp, on=["TEAM", "OPP", "env_key"], how="left")
            df_lines["line_salary_total"] = df_lines["line_salary_total"].fillna(0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                df_lines["line_salary_mean"] = df_lines["line_salary_total"] / df_lines["n_players"].replace(0, np.nan)
                df_lines["line_value_score"] = df_lines["line_score"] / (df_lines["line_salary_total"] / 1000.0).replace(0, np.nan)
        else:
            df_lines["line_salary_total"] = 0.0
            df_lines["line_salary_mean"] = 0.0
            df_lines["line_value_score"] = 0.0
    except Exception as e:
        print(f"Salary aggregation failed: {e}")
        df_lines["line_salary_total"] = df_lines.get("line_salary_total", 0.0)
        df_lines["line_salary_mean"] = df_lines.get("line_salary_mean", 0.0)
        df_lines["line_value_score"] = df_lines.get("line_value_score", 0.0)

    # ----------------------------------------
    # DFS tiers based on line_score
    # ----------------------------------------
    scores = pd.to_numeric(df_lines.get("line_score", np.nan), errors="coerce")
    q80 = float(scores.quantile(0.8)) if len(scores) else 0.0
    q60 = float(scores.quantile(0.6)) if len(scores) else 0.0
    q40 = float(scores.quantile(0.4)) if len(scores) else 0.0

    def tier_func(x: float) -> str:
        if x >= q80:
            return "S"
        elif x >= q60:
            return "A"
        elif x >= q40:
            return "B"
        else:
            return "C"

    df_lines["tier"] = scores.apply(tier_func)

    # Value tiers based on line_value_score
    val_scores = pd.to_numeric(df_lines.get("line_value_score", np.nan), errors="coerce")
    v80 = float(val_scores.quantile(0.8)) if len(val_scores) else 0.0
    v60 = float(val_scores.quantile(0.6)) if len(val_scores) else 0.0

    def value_tier(v: float) -> str:
        if v >= v80:
            return "VALUE_S"
        elif v >= v60:
            return "VALUE_A"
        else:
            return "VALUE_B"

    df_lines["value_tier"] = val_scores.apply(value_tier)

    return df_lines


# ---------------------------------------------------------------------
# Game Classifier
# ---------------------------------------------------------------------
def classify_game(
    df_lines: pd.DataFrame,
    home_team: str,
    away_team: str,
    vegas_total: Optional[float] = None,
) -> Dict[str, Any]:
    home = df_lines[df_lines["TEAM"] == home_team]
    away = df_lines[df_lines["TEAM"] == away_team]

    if home.empty or away.empty:
        return {
            "game": f"{home_team} vs {away_team}",
            "model_total": 0.0,
            "blended_total": 0.0,
            "danger_score": 0.0,
            "classification": "unknown",
        }

    home_xg = home["line_xg_pg"].sum()
    away_xg = away["line_xg_pg"].sum()

    home_goalie_mult = float(home["goalie_mult"].iloc[0])
    away_goalie_mult = float(away["goalie_mult"].iloc[0])

    home_adj = home_xg * away_goalie_mult
    away_adj = away_xg * home_goalie_mult

    model_total = home_adj + away_adj

    if vegas_total is not None:
        blended_total = 0.6 * model_total + 0.4 * vegas_total
    else:
        blended_total = model_total

    home_hd = float(home["line_hdcf"].mean())
    away_hd = float(away["line_hdcf"].mean())

    home_pp1_strength = float(home["pp1_count"].sum())
    away_pp1_strength = float(away["pp1_count"].sum())

    danger_score = (
        0.6 * (home_hd + away_hd)
        + 0.4 * (home_pp1_strength + away_pp1_strength)
    )

    if blended_total >= 6.8 or (model_total >= 6.5 and danger_score >= 2.5):
        classification = "shootout"
    elif blended_total >= 6.2:
        classification = "high"
    elif blended_total >= 5.6:
        classification = "neutral"
    else:
        classification = "low"

    return {
        "game": f"{home_team} vs {away_team}",
        "model_total": round(model_total, 3),
        "blended_total": round(blended_total, 3),
        "danger_score": round(danger_score, 3),
        "classification": classification,
    }


# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------
def main(n_sims: int = 10000) -> None:
    print(f"Loading player projections from {INPUT_FILE}")
    df = load_projections(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")

    vegas_lookup = load_rw_vegas_totals()
    df_matchups = load_matchups_summary()

    print("Building line-level model (expected goals + boom probability)...")
    df_lines = build_line_model(df, df_matchups=df_matchups, n_sims=n_sims)

    # Add game_class and vegas_total
    def classify_row(row):
        vegas_total = vegas_lookup.get(row["TEAM"], None)
        out = classify_game(
            df_lines,
            row["TEAM"],
            row["OPP"],
            vegas_total=vegas_total,
        )
        return out["classification"]

    df_lines["game_class"] = df_lines.apply(classify_row, axis=1)

    def get_game_vegas_total(row):
        vt1 = vegas_lookup.get(row["TEAM"], None)
        vt2 = vegas_lookup.get(row["OPP"], None)
        if vt1 is not None:
            return vt1
        if vt2 is not None:
            return vt2
        return None

    df_lines["vegas_total"] = df_lines.apply(get_game_vegas_total, axis=1)

    print(f"Writing line model to {OUTPUT_FILE}")
    df_lines.to_csv(OUTPUT_FILE, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
