# nhl_projection_engine.py
#
# Builds NHL DFS projections from:
#   - merged_nhl_player_pool.csv (Rotowire + MoneyPuck merged data, incl. recency)
#   - nhl-player-props-overview-book.xlsx (optional wide props export)
#
# Output:
#   - nhl_player_projections.csv with:
#       * base_fpts_model        - blended RW + MoneyPuck (season + recency)
#       * mp_fpts_model          - MoneyPuck season-only model (per-game)
#       * mp_fpts_model_recency  - recency-adjusted MoneyPuck model
#       * prop_adj_fpts          - fantasy-points adjusted via props (small multipliers)
#       * final_fpts             - projection used downstream (nhl_export_for_fd.py, sims, etc.)
#
# Notes:
#   - Rotowire remains the spine (salary, FPTS, Vegas).
#   - Recency is ALWAYS ON and comes from:
#       * xG_per60_recency, SOG_per60_recency, CF_per60_recency
#       * xGF_pg_recency, xGA_pg_recency (team)
#       * xGA_pg_recency_goalie (goalies)
#   - Goalies use a model based on team xGA, goalie xG per game + recency + Win %.
#   - Matchup multipliers are applied ONCE with tight caps to avoid nonsense outliers.

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

APP_ROOT = Path(__file__).parent.resolve()

MERGED_FILE = APP_ROOT / "merged_nhl_player_pool.csv"
PROPS_FILE  = APP_ROOT / "nhl-player-props-overview-book.xlsx"
OUTPUT_FILE = APP_ROOT / "nhl_player_projections.csv"


MERGED_FILE = APP_ROOT / "merged_nhl_player_pool.csv"
PROPS_FILE = APP_ROOT / "nhl-player-props-overview-book.xlsx"  # optional
OUTPUT_FILE = APP_ROOT / "nhl_player_projections.csv"

# --- FanDuel weights (adjust if you use a custom table) ---
FD_GOAL = 12.0
FD_AST  = 8.0
FD_SOG  = 1.6
FD_BLK  = 1.6

# Goalie scoring bits used in compute_goalie_fpts
FD_WIN = 12.0
FD_SHUTOUT = 8.0

# When MoneyPuck assists per-game are missing, approximate from xG
ASSIST_PER_XG = 0.85   # conservative proxy (tunable)

# Props-as-multiplier settings (kept light and clamped)
PROP_MULT_ENABLED = True
BASE_GOAL_PROB = 0.20  # league-ish baseline chance of 1+ goal
GOAL_ALPHA = 0.25      # sensitivity of multiplier to goal prob deviation
SOG_ALPHA  = 0.06      # small nudge from SOG O/U lines
MULT_MIN, MULT_MAX = 0.95, 1.10  # clamps so props never dominate

# Recency blend weights
REC_XG_WEIGHT = 0.7   # how much recency influences xG vs season
REC_SOG_WEIGHT = 0.3  # how much recency influences SOG vs season
REC_MP_WEIGHT = 0.40  # how much of MP model is tilted by recency when blending
REC_RW_WEIGHT = 0.65  # how much RW stays in control in base_fpts_model


def _safe_div(a, b):
    b = b if (b and b != 0) else 1.0
    return a / b


def moneypuck_fpts_row(row):
    """
    Build expected FD points from MoneyPuck player-level stats.
    Uses per-game rates: xG, shots on goal, blocks, and assists (from MP if present,
    otherwise a tunable proxy from xG).
    """
    gp = row.get("games_played") or row.get("games_played_env") or row.get("games") or 0
    gp = gp if gp and gp > 0 else 1.0

    xg   = float(row.get("I_F_xGoals", 0.0))
    sog  = float(row.get("I_F_shotsOnGoal", 0.0))
    blk_attempts = float(row.get("I_F_blockedShotAttempts", 0.0))
    blk  = float(row.get("I_F_blockedShots", 0.0)) if row.get("I_F_blockedShots") is not None else 0.55 * blk_attempts

    # Per-game rates
    xg_pg  = _safe_div(xg, gp)
    sog_pg = _safe_div(sog, gp)
    blk_pg = _safe_div(blk, gp)

    # Assists per game: use MP assists if you have them; otherwise proxy from xG
    pri_ast = float(row.get("I_F_primaryAssists", 0.0))
    sec_ast = float(row.get("I_F_secondaryAssists", 0.0))
    ast_pg_mp = _safe_div(pri_ast + sec_ast, gp)
    ast_pg = ast_pg_mp if ast_pg_mp > 0 else ASSIST_PER_XG * xg_pg

    # Expected FD points
    fpts = (
        FD_GOAL * xg_pg +
        FD_AST  * ast_pg +
        FD_SOG  * sog_pg +
        FD_BLK  * blk_pg
    )
    return max(fpts, 0.0)


# -----------------------------
# Goalie model helpers (with recency)
# -----------------------------
def compute_goalie_fpts(df: pd.DataFrame) -> pd.Series:
    d = df.copy()

    # Team environment stats - use recency if present, otherwise season proxy
    xga_team = pd.to_numeric(
        d.get("xGA_pg_recency", d.get("xGoalsAgainst_teamenv", np.nan)),
        errors="coerce",
    )
    sog_team = pd.to_numeric(
        d.get("shotsOnGoalAgainst_teamenv", d.get("shotsAgainst_teamenv", np.nan)),
        errors="coerce",
    )

    # Handle Series vs scalar safely for league baselines
    if isinstance(xga_team, pd.Series):
        league_xga = xga_team.replace(0, np.nan).mean()
    else:
        league_xga = np.nan

    if isinstance(sog_team, pd.Series):
        league_sog = sog_team.replace(0, np.nan).mean()
    else:
        league_sog = np.nan

    if not np.isfinite(league_xga) or league_xga <= 0:
        league_xga = 2.8
    if not np.isfinite(league_sog) or league_sog <= 0:
        league_sog = 30.0

    # Normalize xga_team / sog_team to Series aligned with df
    if not isinstance(xga_team, pd.Series):
        xga_team = pd.Series(league_xga, index=d.index)
    if not isinstance(sog_team, pd.Series):
        sog_team = pd.Series(league_sog, index=d.index)

    xga_team = xga_team.fillna(league_xga)
    sog_team = sog_team.fillna(league_sog)

    # Goalie personal stats
    g_xg = pd.to_numeric(d.get("goalie_xGoals", np.nan), errors="coerce")
    g_sog = pd.to_numeric(d.get("goalie_shotsOnGoal", np.nan), errors="coerce")
    g_goals = pd.to_numeric(d.get("goalie_goals", np.nan), errors="coerce")
    g_games = pd.to_numeric(d.get("goalie_games_played", np.nan), errors="coerce")

    # Season-level goalie xGA per game
    g_xg_pg_season = np.where(
        (g_xg > 0) & (g_games > 0),
        g_xg / g_games,
        np.nan,
    )

    # Recency xGA per game if available
    g_xg_pg_rec = pd.to_numeric(d.get("xGA_pg_recency_goalie", np.nan), errors="coerce")

    # Blend recency and season; fallback to team env if we have nothing
    g_xg_per_game = np.where(
        np.isfinite(g_xg_pg_rec),
        0.70 * g_xg_pg_rec + 0.30 * np.where(
            np.isfinite(g_xg_pg_season),
            g_xg_pg_season,
            xga_team,
        ),
        np.where(
            np.isfinite(g_xg_pg_season),
            g_xg_pg_season,
            xga_team,
        ),
    )

    # Per-game shots against: use goalie if we have it, else team
    g_sog_per_game = np.where(
        (g_sog > 0) & (g_games > 0),
        g_sog / g_games,
        sog_team,
    )

    # Simple save% model and expected goals against
    save_pct = np.where(
        g_sog_per_game > 0,
        1.0 - (g_xg_per_game / g_sog_per_game),
        0.910,
    )
    save_pct = np.clip(save_pct, 0.880, 0.940)

    exp_ga = g_xg_per_game
    exp_saves = g_sog_per_game * save_pct

    # Win % from Rotowire if present
    win_pct = pd.to_numeric(d.get("Win %", np.nan), errors="coerce")
    if not np.isfinite(win_pct).any():
        win_pct = 0.5  # neutral fallback
    else:
        if win_pct.max() > 1.0:
            win_pct = win_pct / 100.0

    # Very rough shutout odds: smaller for higher xGA
    shutout_pct = np.clip(0.12 - 0.02 * (exp_ga - 2.0), 0.01, 0.15)

    fpts = (
        exp_saves * FD_SOG
        - exp_ga * 4.0
        + win_pct * FD_WIN
        + shutout_pct * FD_SHUTOUT
    )

    return pd.Series(fpts, index=d.index)


# -----------------------------
# Props / Vegas helpers
# -----------------------------
def attach_props(df: pd.DataFrame, props_file: Path) -> pd.DataFrame:
    d = df.copy()

    if not props_file.exists():
        print(" Props file not found; skipping external prop merge.")
        return d

    try:
        _ = pd.read_excel(props_file, sheet_name=None)
    except Exception as e:
        print(f" Failed to read props file ({e}); skipping.")
        return d

    # wide overview - user-specific / book-specific layout; we'll gently pick
    # columns if present; otherwise we won't override anything
    # For now we rely primarily on Rotowire + inline "1+ G", "1+ PTS", "O/U SOG"
    return d


# ----------------------------
# Props as a gentle multiplier (optional)
# ----------------------------
def _props_multiplier(row):
    if not PROP_MULT_ENABLED:
        return 1.0

    mult = 1.0

    # Goal probability bump (column often like "1+ G" with decimal prob or %)
    p_goal_raw = row.get("1+ G", None)
    if p_goal_raw is not None:
        try:
            p_goal = float(p_goal_raw)
            p_goal = p_goal / 100.0 if p_goal > 1.0 else p_goal  # handle % inputs
            mult *= 1.0 + GOAL_ALPHA * (p_goal - BASE_GOAL_PROB)
        except Exception:
            pass

    # SOG line bump (column often like "O/U SOG" with numeric prop line)
    sog_line = row.get("O/U SOG", None)
    if sog_line is not None:
        try:
            sog_line = float(sog_line)
            # Compare to our SOG per game estimate
            gp = row.get("games_played") or 1.0
            sog_pg = _safe_div(float(row.get("I_F_shotsOnGoal", 0.0)), gp)
            mult *= 1.0 + SOG_ALPHA * (sog_line - sog_pg)
        except Exception:
            pass

    # Clamp so props never overpower the model
    return max(MULT_MIN, min(MULT_MAX, mult))


def compute_prop_adjusted_fpts(d: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative props application:
      prop_adj_fpts = base_fpts_model * prop_mult
    Where prop_mult comes from _props_multiplier() with tight clamps.
    """
    d = d.copy()
    # Build the multiplier here (safe even if props are missing; returns 1.0)
    try:
        d["prop_mult"] = d.apply(_props_multiplier, axis=1)
    except Exception:
        d["prop_mult"] = 1.0

    base = pd.to_numeric(d.get("base_fpts_model", 0.0), errors="coerce").fillna(0.0)
    mult = pd.to_numeric(d.get("prop_mult", 1.0), errors="coerce").fillna(1.0)
    d["prop_adj_fpts"] = (base * mult).clip(lower=0.0)
    return d


# -----------------------------
# Main projection computation
# -----------------------------
def compute_fantasy_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build:
      - base_fpts_model        - blended RW + MoneyPuck (season + recency)
      - mp_fpts_model          - MoneyPuck season-only model
      - mp_fpts_model_recency  - recency-adjusted MoneyPuck model
      - prop_adj_fpts          - from Vegas / external props (conservative)
      - final_fpts             - blended projection (props + Rotowire + goalie model)
    """
    d = df.copy()

    # Rotowire FPTS as our baseline "model"
    if "FPTS" in d.columns:
        d["rw_fpts"] = pd.to_numeric(d["FPTS"], errors="coerce").fillna(0.0)
    else:
        d["rw_fpts"] = 0.0

    # ---------- MoneyPuck season-based model ----------
    gp = pd.to_numeric(d.get("games_played", np.nan), errors="coerce")
    gp = gp.where((gp > 0) & np.isfinite(gp), np.nan)  # only valid gp

    xg_tot   = pd.to_numeric(d.get("I_F_xGoals", np.nan), errors="coerce")
    sog_tot  = pd.to_numeric(d.get("I_F_shotsOnGoal", np.nan), errors="coerce")
    g_tot    = pd.to_numeric(d.get("goals", np.nan), errors="coerce")
    pts_tot  = pd.to_numeric(d.get("points", np.nan), errors="coerce")

    xg_pg   = (xg_tot / gp).where(gp.notna(), np.nan)
    sog_pg  = (sog_tot / gp).where(gp.notna(), np.nan)
    g_pg    = (g_tot / gp).where(gp.notna(), np.nan)
    pts_pg  = (pts_tot / gp).where(gp.notna(), np.nan)

    ast_pg = np.clip(pts_pg - g_pg, 0.0, None)

    mp_fpts_pg = (
        FD_GOAL * xg_pg.fillna(0.0) +
        FD_AST  * ast_pg.fillna(0.0) +
        FD_SOG  * sog_pg.fillna(0.0)
    )

    # Fallback: per-60 * avg TOI if per-game is unavailable
    xg60  = pd.to_numeric(d.get("xG_per60", np.nan), errors="coerce")
    sog60 = pd.to_numeric(d.get("shots_on_goal_per60", d.get("SOG_per60", np.nan)), errors="coerce")
    toi_tot = pd.to_numeric(d.get("icetime", np.nan), errors="coerce")  # season seconds
    avg_toi = (toi_tot / gp).where((toi_tot > 0) & gp.notna(), np.nan)

    # ---- SAFETY PATCH: xg60 and avg_toi must both be Series ----
    # xg60 recency fallback earlier may produce a float if missing
    xg60 = pd.to_numeric(xg60, errors="coerce")
    if not isinstance(xg60, pd.Series):
        # Convert single float to repeated Series
        xg60 = pd.Series([xg60] * len(df), index=df.index)
    
    # avg_toi may break if icetime/games_played gets cast to scalar
    avg_toi = pd.to_numeric(avg_toi, errors="coerce")
    if not isinstance(avg_toi, pd.Series):
        avg_toi = pd.Series([avg_toi] * len(df), index=df.index)
    
    # Now safe to use fillna

    mp_fpts_alt = (
        FD_GOAL * xg60.fillna(0.0) * avg_toi.fillna(0.0) / 60.0 +
        FD_SOG  * sog60.fillna(0.0) * avg_toi.fillna(0.0) / 60.0
    )

    mp_fpts_model = mp_fpts_pg.where(mp_fpts_pg.notna(), mp_fpts_alt)
    d["mp_fpts_model"] = pd.to_numeric(mp_fpts_model, errors="coerce").fillna(0.0)

    # ---------- Recency-adjusted MoneyPuck model ----------
    xg60_season = pd.to_numeric(d.get("xG_per60", np.nan), errors="coerce")
    xg60_rec    = pd.to_numeric(d.get("xG_per60_recency", np.nan), errors="coerce")

    sog60_season = pd.to_numeric(d.get("SOG_per60", d.get("shots_on_goal_per60", np.nan)), errors="coerce")
    sog60_rec    = pd.to_numeric(d.get("SOG_per60_recency", np.nan), errors="coerce")

    # Ratios: recency vs season, softly clipped
    with np.errstate(divide="ignore", invalid="ignore"):
        xg_ratio = np.where(
            (xg60_season > 0) & np.isfinite(xg60_season) & np.isfinite(xg60_rec),
            xg60_rec / xg60_season,
            1.0,
        )
        sog_ratio = np.where(
            (sog60_season > 0) & np.isfinite(sog60_season) & np.isfinite(sog60_rec),
            sog60_rec / sog60_season,
            1.0,
        )

    xg_ratio  = np.clip(xg_ratio, 0.60, 1.50)
    sog_ratio = np.clip(sog_ratio, 0.70, 1.40)

    rec_factor = (
        REC_XG_WEIGHT * xg_ratio +
        REC_SOG_WEIGHT * sog_ratio
    )
    rec_factor = np.clip(rec_factor, 0.70, 1.40)

    mp_fpts_rec = d["mp_fpts_model"] * rec_factor
    d["mp_fpts_model_recency"] = pd.to_numeric(mp_fpts_rec, errors="coerce").fillna(d["mp_fpts_model"])

    # Blended MP model: 60% season, 40% recency-tilted
    mp_blend = 0.60 * d["mp_fpts_model"] + REC_MP_WEIGHT * d["mp_fpts_model_recency"]
    d["mp_fpts_model_blend"] = mp_blend

    # ---- Base model: RW + MoneyPuck (season+recency) ----
    rw = d["rw_fpts"].fillna(0.0)
    mp = d["mp_fpts_model_blend"].fillna(0.0)
    d["base_fpts_model"] = REC_RW_WEIGHT * rw + (1.0 - REC_RW_WEIGHT) * mp

    # -----------------------------
    # Opponent matchup adjustment (team-level, recency-aware)
    # -----------------------------
    opp_def = pd.to_numeric(
        d.get("xGA_pg_recency_opp", d.get("xGoalsAgainst_teamenv_opp", np.nan)),
        errors="coerce",
    )
    opp_sog = pd.to_numeric(
        d.get("shotsOnGoalAgainst_teamenv_opp", np.nan),
        errors="coerce",
    )

    if not isinstance(opp_def, pd.Series):
        opp_def = pd.Series(opp_def, index=d.index)
    if not isinstance(opp_sog, pd.Series):
        opp_sog = pd.Series(opp_sog, index=d.index)

    league_xga = opp_def.replace(0, np.nan).mean()
    league_sog = opp_sog.replace(0, np.nan).mean()

    if not np.isfinite(league_xga) or league_xga <= 0:
        league_xga = 2.8
    if not np.isfinite(league_sog) or league_sog <= 0:
        league_sog = 30.0

    def_mult = np.where(
        np.isfinite(opp_def),
        np.clip(opp_def / league_xga, 0.80, 1.20),
        1.0,
    )
    sog_mult = np.where(
        np.isfinite(opp_sog),
        np.clip(opp_sog / league_sog, 0.80, 1.20),
        1.0,
    )

    matchup_mult = np.clip(
        0.60 * def_mult + 0.40 * sog_mult,
        0.85,
        1.15,
    )
    d["matchup_mult"] = matchup_mult

    # -----------------------------
    # Optional 5v5 line-vs-line matchup multiplier (summary-first)
    # -----------------------------
    line_matchup_col = "line_matchup_mult"
    d[line_matchup_col] = 1.0  # neutral by default

    FIVEV5_SUMMARY_CANDIDATES = [
        APP_ROOT / "5v5_matchups_summary.csv",
        APP_ROOT / "5v5_matchups.csv",
        Path("5v5_matchups_summary.csv"),
        Path("5v5_matchups.csv"),
    ]

    def _normalize_line_tag(x):
        s = str(x).upper().strip()
        if s in {"1", "2", "3", "4"}:
            return f"FL{s}"
        if s.startswith("FL"):
            return s
        if s.startswith("L") and s[1:].isdigit():
            return f"FL{s[1:]}"
        return s

    def _load_5v5_summary() -> Optional[pd.DataFrame]:
        for p in FIVEV5_SUMMARY_CANDIDATES:
            try:
                if p.exists():
                    m = pd.read_csv(p)
                    cols = {c.lower(): c for c in m.columns}
                    req = {"team", "line", "opp_team", "opp_line", "toi"}
                    if not req.issubset(set(cols.keys())):
                        continue
                    m = m.rename(
                        columns={
                            cols["team"]: "team",
                            cols["line"]: "line",
                            cols["opp_team"]: "opp_team",
                            cols["opp_line"]: "opp_line",
                            cols["toi"]: "toi",
                        }
                    )
                    m["team"] = m["team"].astype(str).str.strip().str.upper()
                    m["line"] = m["line"].astype(str).str.strip().str.upper().map(_normalize_line_tag)
                    m["opp_team"] = m["opp_team"].astype(str).str.strip().str.upper()
                    m["opp_line"] = m["opp_line"].astype(str).str.strip().str.upper().map(_normalize_line_tag)
                    m["toi"] = pd.to_numeric(m["toi"], errors="coerce").fillna(0.0)
                    return m
            except Exception:
                continue
        return None

    m = _load_5v5_summary()
    if m is not None:
        m = m.copy()
        total_toi = m.groupby(["team", "line"])["toi"].transform("sum")
        m["weight"] = np.where(total_toi > 0, m["toi"] / total_toi, 0.0)

        # Build per-team/line offensive & defensive strengths from current DF
        k = d.copy()
        team_col = "TEAM" if "TEAM" in k.columns else "team"

        if "LINE" in k.columns:
            k["LINE_KEY"] = k["LINE"].astype(str).str.strip().str.upper().map(_normalize_line_tag)
        elif "line" in k.columns:
            k["LINE_KEY"] = k["line"].astype(str).str.strip().str.upper().map(_normalize_line_tag)
        elif "line_num" in k.columns:
            k["LINE_KEY"] = k["line_num"].apply(lambda n: f"FL{int(n)}" if pd.notna(n) else np.nan)
        else:
            k["LINE_KEY"] = np.nan

        k["TEAM_KEY"] = k[team_col].astype(str).str.strip().str.upper()

        line_off_vals = pd.to_numeric(k.get("line_strength_raw", np.nan), errors="coerce")
        if not np.isfinite(line_off_vals).any():
            line_off_vals = pd.to_numeric(
                k.get("line_offense_mult", 1.0), errors="coerce"
            ).fillna(1.0)

        line_def_vals = pd.to_numeric(
            k.get("line_defense_mult", np.nan), errors="coerce"
        )
        team_def_vals = pd.to_numeric(
            k.get("team_defense_mult", 1.0), errors="coerce"
        ).fillna(1.0)

        k["__line_off"] = line_off_vals
        k["__line_def"] = line_def_vals
        k["__team_def"] = team_def_vals

        agg = (
            k.groupby(["TEAM_KEY", "LINE_KEY"])[["__line_off", "__line_def", "__team_def"]]
            .mean()
            .reset_index()
        )
        agg = agg.rename(
            columns={
                "TEAM_KEY": "team",
                "LINE_KEY": "line",
                "__line_off": "line_off",
                "__line_def": "line_def",
                "__team_def": "team_def",
            }
        )

        # Attach offensive strength for our line
        m = m.merge(agg[["team", "line", "line_off"]], on=["team", "line"], how="left")

        # Build opponent defense table (line + team)
        opp_agg = agg.rename(
            columns={
                "team": "opp_team",
                "line": "opp_line",
                "line_def": "opp_line_def",
                "team_def": "opp_team_def",
            }
        )[["opp_team", "opp_line", "opp_line_def", "opp_team_def"]]

        m = m.merge(opp_agg, on=["opp_team", "opp_line"], how="left")

        opp_line_def = pd.to_numeric(m.get("opp_line_def", np.nan), errors="coerce")
        opp_team_def = pd.to_numeric(m.get("opp_team_def", 1.0), errors="coerce").fillna(1.0)
        eff_def = np.where(
            np.isfinite(opp_line_def),
            opp_line_def * opp_team_def,
            opp_team_def,
        )

        off_val = pd.to_numeric(m.get("line_off", np.nan), errors="coerce")
        ratio = np.where(
            (eff_def > 0)
            & np.isfinite(eff_def)
            & np.isfinite(off_val)
            & (off_val > 0),
            np.clip(off_val / eff_def, 0.80, 1.20),
            1.0,
        )

        m["value"] = m["weight"] * ratio

        lm = m.groupby(["team", "line"])["value"].sum().reset_index()
        lm[line_matchup_col] = lm["value"].clip(0.85, 1.15)

        # Merge back onto player DF
        d = d.copy()
        if "TEAM_KEY" not in d.columns:
            d["TEAM_KEY"] = k["TEAM_KEY"]
        if "LINE_KEY" not in d.columns:
            d["LINE_KEY"] = k["LINE_KEY"]

        if line_matchup_col in d.columns:
            d.drop(columns=[line_matchup_col], inplace=True, errors="ignore")

        d = d.merge(
            lm[["team", "line", line_matchup_col]],
            left_on=["TEAM_KEY", "LINE_KEY"],
            right_on=["team", "line"],
            how="left",
            suffixes=("", "_right"),
        )

        if f"{line_matchup_col}_right" in d.columns:
            d[line_matchup_col] = d[f"{line_matchup_col}_right"]
            d.drop(columns=[f"{line_matchup_col}_right"], inplace=True, errors="ignore")

        if line_matchup_col not in d.columns:
            d[line_matchup_col] = 1.0
        d[line_matchup_col] = pd.to_numeric(d[line_matchup_col], errors="coerce").fillna(1.0)

        d.drop(
            columns=["TEAM_KEY", "LINE_KEY", "team", "line"],
            inplace=True,
            errors="ignore",
        )
    else:
        print(" LvL: no 5v5 file found; using neutral line multiplier.")
        if line_matchup_col not in d.columns:
            d[line_matchup_col] = 1.0

    # Combined team + line matchup metric (for reference)
    d["final_matchup_mult"] = d.get("matchup_mult", 1.0) * d[line_matchup_col]

    # -------------------------------
    # Line Strength (Normalized)
    # -------------------------------
    if "line_strength_raw" in d.columns:
        raw = pd.to_numeric(d["line_strength_raw"], errors="coerce")
        span = raw.max() - raw.min()
        if pd.notna(span) and span > 0:
            line_norm = (raw - raw.min()) / span
        else:
            line_norm = pd.Series(0.5, index=d.index)  # neutral
        d["line_strength_norm"] = (line_norm * 100.0).fillna(50.0)

        # Build a SMALL multiplier from line strength: ~[-5%, +5%]
        line_strength_mult = 0.95 + (line_norm.clip(0.0, 1.0) * 0.10)
        d["line_strength_mult"] = line_strength_mult.clip(0.95, 1.05)
        d["line_matchup_strength"] = d["line_strength_norm"]
    else:
        d["line_strength_norm"] = 50.0
        d["line_strength_mult"] = 1.0
        d["line_matchup_strength"] = 50.0

    # Prop-based estimate (Vegas-first, then external props)
    d = compute_prop_adjusted_fpts(d)

    prop = d["prop_adj_fpts"].fillna(0.0)
    rw = d["rw_fpts"].fillna(0.0)

    # Goalie vs skater split
    pos = d["base_pos"].astype(str).str.upper()
    is_goalie = pos.eq("G")
    is_skater = ~is_goalie

    # Skaters: conservative blend Rotowire + props on top of base_fpts_model
    skater_final = np.where(
        (prop > 0) & (rw > 0),
        0.70 * d["base_fpts_model"] + 0.30 * prop,
        np.where(prop > 0, prop, d["base_fpts_model"]),
    )

    final = np.zeros(len(d), dtype=float)
    final[is_skater] = skater_final[is_skater]

    # Goalies: use special goalie model + RW
    if is_goalie.any():
        goalie_df = d.loc[is_goalie].copy()
        goalie_model = compute_goalie_fpts(goalie_df)
        rw_g = rw[is_goalie]

        goalie_final = np.where(
            (goalie_model > 0) & (rw_g > 0),
            0.60 * goalie_model + 0.40 * rw_g,
            np.where(goalie_model > 0, goalie_model, rw_g),
        )

        final[is_goalie] = goalie_final

    # ------------------------------------------------------------
    # SAFE MATCHUP MULTIPLIERS (apply ONCE with tight caps)
    # ------------------------------------------------------------
    team_mult = pd.to_numeric(d.get("matchup_mult", 1.0), errors="coerce").fillna(1.0)
    line_mult = pd.to_numeric(d.get(line_matchup_col, 1.0), errors="coerce").fillna(1.0)
    strength_mult = pd.to_numeric(d.get("line_strength_mult", 1.0), errors="coerce").fillna(1.0)

    team_mult = team_mult.clip(0.85, 1.15)
    line_mult = line_mult.clip(0.85, 1.15)
    strength_mult = strength_mult.clip(0.95, 1.05)

    combined_mult = (team_mult * line_mult * strength_mult).clip(0.80, 1.25)

    final = final * combined_mult

    # ------------------------------------------------------------
    # Sanity rescale: if slate is globally inflated, bring back to DFS range
    # ------------------------------------------------------------
    positive = final[(final > 0) & np.isfinite(final)]
    if len(positive) > 0:
        median_fpts = float(np.median(positive))
    else:
        median_fpts = np.nan

    # Target a median around ~25 FD points for main scorers.
    if np.isfinite(median_fpts) and median_fpts > 30.0:
        scale = 25.0 / median_fpts
        print(
            f" Global projection rescale applied. "
            f"Median={median_fpts:.2f} -> scale factor={scale:.3f}"
        )
        final = final * scale
        d["projection_rescale_factor"] = scale
    else:
        d["projection_rescale_factor"] = 1.0

    d["final_fpts"] = pd.to_numeric(final, errors="coerce").fillna(0.0)

    # -------------------------------
    # Player Volatility (standard deviation)
    # -------------------------------
    base = d["final_fpts"].clip(lower=0.1)

    # Position flags
    pos = d["base_pos"].astype(str).str.upper()
    is_goalie = pos.eq("G")
    is_skater = ~is_goalie

    # Reuse season totals and games-played from earlier
    gp_safe = gp.where((gp > 0) & np.isfinite(gp), np.nan)

    xg_pg_vol   = (xg_tot / gp_safe).where(gp_safe.notna(), np.nan).fillna(0.0)
    sog_pg_vol  = (sog_tot / gp_safe).where(gp_safe.notna(), np.nan).fillna(0.0)
    pts_pg_vol  = (pts_tot / gp_safe).where(gp_safe.notna(), np.nan).fillna(0.0)
    g_pg_vol    = (g_tot / gp_safe).where(gp_safe.notna(), np.nan).fillna(0.0)
    ast_pg_vol  = np.clip(pts_pg_vol - g_pg_vol, 0.0, None)

    blk_attempts_raw = d.get("I_F_blockedShotAttempts", np.nan)
    blk_made_raw     = d.get("I_F_blockedShots", np.nan)

    blk_attempts_tot = pd.to_numeric(pd.Series(blk_attempts_raw), errors="coerce").fillna(0.0)
    blk_tot          = pd.to_numeric(pd.Series(blk_made_raw), errors="coerce")
    blk_tot = blk_tot.where(blk_tot.notna(), 0.55 * blk_attempts_tot)
    blk_pg_vol = (blk_tot / gp_safe).where(gp_safe.notna(), np.nan).fillna(0.0)

    # --- Event-level sigmas in fantasy points ---
    sigma_goals_fp = np.sqrt(xg_pg_vol.clip(lower=0.0)) * FD_GOAL
    sigma_sog_fp   = np.sqrt(sog_pg_vol.clip(lower=0.0)) * (FD_SOG * 0.9)
    sigma_ast_fp   = np.sqrt(ast_pg_vol.clip(lower=0.0)) * (FD_AST * 0.7)
    sigma_blk_fp   = np.sqrt(blk_pg_vol.clip(lower=0.0)) * (FD_BLK * 0.6)

    avg_toi_safe = pd.to_numeric(avg_toi, errors="coerce").fillna(0.0)
    sigma_toi_fp = (avg_toi_safe / 60.0) * base * 0.35

    combined_mult_safe = pd.to_numeric(combined_mult, errors="coerce").fillna(1.0)
    sigma_env_fp = np.abs(combined_mult_safe - 1.0) * base

    raw_sigma = np.sqrt(
        sigma_goals_fp**2
        + sigma_sog_fp**2
        + sigma_ast_fp**2
        + sigma_blk_fp**2
        + sigma_toi_fp**2
        + sigma_env_fp**2
    )

    pos_scale = np.where(
        pos.eq("D"), 0.85,
        np.where(pos.eq("W"), 1.10, 1.00),
    )

    sigma_all = raw_sigma * pos_scale
    sigma = pd.Series(sigma_all, index=d.index)

    # Goalies: override volatility
    sigma.loc[is_goalie] = np.maximum(base[is_goalie] * 0.7, 8.0)

    d["fpts_sigma"] = pd.to_numeric(sigma, errors="coerce").fillna(0.1)

    # -----------------------------------------------------------
    # Goalie volatility override (DFS realism)
    # -----------------------------------------------------------
    is_goalie = d["base_pos"].astype(str).str.upper().eq("G")
    is_skater = ~is_goalie

    if is_goalie.any():
        raw_base = d["final_fpts"].clip(lower=0.1)
        goalie_sigma = 0.45 * raw_base + 0.20 * raw_base + 4.0
        d.loc[is_goalie, "fpts_sigma"] = np.clip(goalie_sigma, 5.0, 15.0)

    # -----------------------------------------------
    # Adaptive slate-based sigma normalization (skaters)
    # -----------------------------------------------
    raw = d.loc[is_skater, "fpts_sigma"].clip(lower=0.01)
    TARGET_MEAN = 2.0
    TARGET_MAX  = 5.5

    divisor = raw.mean() / TARGET_MEAN if raw.mean() > 0 else 1.0
    divisor = max(divisor, 1.0)

    norm = raw / divisor
    norm = norm.clip(lower=0.6, upper=TARGET_MAX)

    d.loc[is_skater, "fpts_sigma"] = norm

    return d


# -----------------------------
# Main runner
# -----------------------------
def run_engine(
    merged_file: Path = MERGED_FILE,
    props_file: Path = PROPS_FILE,
    output_file: Path = OUTPUT_FILE,
):
    print(f" Reading merged player pool -> {merged_file}")
    df = pd.read_csv(merged_file)

    print(f" Attaching props (if available) -> {props_file}")
    df = attach_props(df, props_file)

    print(" Computing fantasy points (base, prop-adjusted, final) with recency...")
    df = compute_fantasy_points(df)

    print(f" Writing projections -> {output_file}")
    df.to_csv(output_file, index=False)
    print(" Done.")


if __name__ == "__main__":
    run_engine()
