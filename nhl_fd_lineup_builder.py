# nhl_fd_lineup_builder.py
#
# FanDuel Classic NHL Lineup Builder (projection-weighted + correlation + matchup + slate sim)
#
# Uses:
#   - nhl_fd_projections.csv (from nhl_export_for_fd.py)
#
# Roster:
#   - 2 C  (Centers)
#   - 4 W  (Wingers)
#   - 2 D  (Defensemen)
#   - 1 G  (Goalie)
#
# Stacking / correlation rules (GPP-oriented):
#   - At least TWO different teams with >= 2 skaters each
#   - Max 3 skaters per team
#   - Goalie rules:
#       * No skaters from goalie's own team
#       * Max 1 skater vs goalie opponent
#   - At least 2 PP1 skaters (if PP info exists)
#   - Max 3 skaters from the same even-strength line (L1/L2/L3/L4)
#
# Search / scoring:
#   - Projection-weighted random sampling under constraints
#   - SMART_SCORE = TOTAL_PROJ + CORR_WEIGHT * CORR_SCORE
#   - CORR_SCORE rewards:
#       * team stacks
#       * line stacks
#       * PP1 stacks
#       * high-projection / good-matchup teams (opponent-driven via PROJ)
#
# Slate simulation:
#   - For each final lineup, simulate N_SIMS slates with player-level randomness
#   - Add SIM_MEAN, SIM_STD, SIM_P75, SIM_P90, SIM_P95, BOOM_PCT columns
#
# NOTE:
#   This builder trusts the PROJ column from nhl_fd_projections.csv.
#   Projection logic (including opponent adjustments, team_env, etc.)
#   lives in nhl_projection_engine.py + nhl_export_for_fd.py.

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import random


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
APP_ROOT = Path(__file__).parent.resolve()
PROJECT_DIR = APP_ROOT


INPUT_PROJ_FILE = PROJECT_DIR / "nhl_fd_projections.csv"
OUTPUT_LINEUPS_FILE = PROJECT_DIR / "nhl_fd_lineups.csv"


# ---------------------------------------------------------------------
# Lineup / sim config
# ---------------------------------------------------------------------
SALARY_CAP = 55000
NUM_LINEUPS = 1000
ITERATIONS = 200000          # number of random attempts
RNG_SEED = 42
CORR_WEIGHT = 1.0          # weight for correlation in SMART_SCORE
N_SIMS = 10000              # slate simulations per lineup (can lower to speed up)

# Simple per-position volatility multipliers for slate sim
VOL_FACTORS = {
    "C": 0.9,
    "W": 0.9,
    "D": 0.7,
    "G": 0.8,
}

# -----------------------------------------
# Stack EV constants
# -----------------------------------------
STACK_LINE_SHARES = {
    "1": 0.40,
    "2": 0.30,
    "3": 0.20,
    "4": 0.10,
    "L1": 0.40,
    "L2": 0.30,
    "L3": 0.20,
    "L4": 0.10,
}

def _estimate_team_ev(team_rows: pd.DataFrame) -> float:
    """Expected goals via your projections model."""
    avg_proj = float(team_rows["PROJ"].mean())
    if not np.isfinite(avg_proj) or avg_proj <= 0:
        return 1.8  # fallback baseline team total
    return float(avg_proj / 12.0)

def _estimate_line_ev(team_ev: float, line_val: str) -> float:
    s = STACK_LINE_SHARES.get(str(line_val).upper(), 0.10)
    return float(team_ev * s)

def _compute_stack_ev(skaters: pd.DataFrame) -> float:
    """Compute blow-up expectation for the best TEAM+LINE cluster in lineup."""
    if skaters.empty:
        return 0.0

    if "LINE" not in skaters.columns:
        # No line info -> fallback: treat team cluster
        team_ev_map = {}
        for team in skaters["TEAM"].unique():
            team_rows = skaters[skaters["TEAM"] == team]
            team_ev_map[team] = _estimate_team_ev(team_rows)
        best = max(team_ev_map.values())
        return best

    # Build team EV
    team_ev_map = {}
    for team in skaters["TEAM"].unique():
        team_rows = skaters[skaters["TEAM"] == team]
        team_ev_map[team] = _estimate_team_ev(team_rows)

    best_line_ev = 0.0
    for (team, line), grp in skaters.groupby(["TEAM", "LINE"]):
        team_ev = team_ev_map.get(team, 1.8)
        line_ev = _estimate_line_ev(team_ev, line)
        if line_ev > best_line_ev:
            best_line_ev = line_ev

    # Convert to GPP "boom chance" via logistic curve
    x = (best_line_ev - 0.65) * 4.0
    boom_chance = 1 / (1 + np.exp(-x))
    return float(boom_chance)


# ---------------------------------------------------------------------
# Load + normalize projections
# ---------------------------------------------------------------------
def load_fd_projections(path: Path) -> pd.DataFrame:
    """Load FD projections and normalize POS, SAL, PROJ, LINE, and PP_LINE.
    Ensures new recency-aware projections remain compatible with the builder.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing nhl_fd_projections.csv: {path}")

    df = pd.read_csv(path)

    # ----------------------------
    # Basic required columns
    # ----------------------------
    for c in ["PLAYER", "TEAM", "OPP"]:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' in projections file.")

    # ----------------------------
    # POS normalization
    # ----------------------------
    pos_source = "POS" if "POS" in df.columns else "base_pos"
    df[pos_source] = df[pos_source].astype(str)

    def normalize_pos(x: str) -> Optional[str]:
        x = x.upper().strip()
        if x in ["LW", "RW", "W", "WING"]:
            return "W"
        if x in ["C", "CENTER", "CENTRE"]:
            return "C"
        if x in ["D", "DEF", "DEFENCE"]:
            return "D"
        if x in ["G", "GOALIE"]:
            return "G"
        return None

    df["POS"] = df[pos_source].apply(normalize_pos)
    df = df[df["POS"].notna()]

    # ----------------------------
    # Salary
    # ----------------------------
    sal_col = "SAL" if "SAL" in df.columns else "Salary"
    df["SAL"] = pd.to_numeric(df[sal_col], errors="coerce").fillna(0).astype(int)

    # ----------------------------
    # PROJ field
    # ----------------------------
    if "PROJ" in df.columns:
        df["PROJ"] = pd.to_numeric(df["PROJ"], errors="coerce").fillna(0.0)
    else:
        fallback = None
        for c in ["final_fpts", "prop_adj_fpts", "FPTS", "base_fpts_model"]:
            if c in df.columns:
                fallback = c
                break
        if fallback is None:
            raise ValueError("Could not find PROJ or fallback projection field.")
        df["PROJ"] = pd.to_numeric(df[fallback], errors="coerce").fillna(0.0)

    # ----------------------------
    # BUILD LINE COLUMN
    # (this is critical — builder logic expects LINE)
    # ----------------------------
    if "LINE" in df.columns:
        pass
    elif "line_num" in df.columns:
        df["LINE"] = df["line_num"].apply(lambda x: f"L{int(x)}" if pd.notna(x) else None)
    elif "env_key" in df.columns:
        # env_key like: "COL_L1"
        df["LINE"] = (
            df["env_key"]
            .astype(str)
            .str.extract(r"_L(\d+)", expand=False)
            .apply(lambda x: f"L{x}" if pd.notna(x) else None)
        )
    else:
        df["LINE"] = None

    # ----------------------------
    # BUILD PP_LINE COLUMN
    # ----------------------------
    if "PP_LINE" in df.columns:
        pass
    elif "pp_unit" in df.columns:
        df["PP_LINE"] = df["pp_unit"].astype(str).str.replace("PP", "")
    elif "PP LINE" in df.columns:
        df["PP_LINE"] = df["PP LINE"]
    else:
        df["PP_LINE"] = None

    # ----------------------------
    # Clean data
    # ----------------------------
    df = df[df["PROJ"] > 0]

    base_cols = ["PLAYER", "TEAM", "OPP", "POS", "SAL", "PROJ", "LINE", "PP_LINE"]
    extra = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + extra].reset_index(drop=True)

    return df

# ---------------------------------------------------------------------
# Stack / goalie rules
# ---------------------------------------------------------------------
def _check_team_stack_rules(teams: List[str]) -> bool:
    """Team stacking: at least two teams with >=2 skaters, max 3 skaters per team."""
    from collections import Counter

    cnt = Counter(teams)
    if any(v > 3 for v in cnt.values()):
        return False

    num_two_plus = sum(1 for v in cnt.values() if v >= 2)
    return num_two_plus >= 2


def _check_goalie_rules(goalie_team: str, goalie_opp: str, skaters: pd.DataFrame) -> bool:
    """Goalie rules: no teammates, max 1 skater vs goalie."""
    if skaters.empty:
        return False

    team_counts = skaters["TEAM"].value_counts()

    if goalie_team in team_counts and team_counts[goalie_team] > 0:
        return False

    if goalie_opp in team_counts and team_counts[goalie_opp] > 1:
        return False

    return True


def _get_line_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["LINE", "Line", "line_num", "EV_LINE", "EV Line"]:
        if c in df.columns:
            return c
    return None


def _get_pp_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["PP LINE", "PP_LINE", "pp_unit", "PP_UNIT"]:
        if c in df.columns:
            return c
    return None


def _check_pp1_stack(skaters: pd.DataFrame) -> bool:
    """Require at least 2 skaters from PP1 (if PP info exists)."""
    col = _get_pp_col(skaters)
    if not col:
        # No PP info -> accept lineup
        return True

    vals = skaters[col].astype(str).str.upper()
    pp1_mask = vals.isin(["1", "PP1"])
    return pp1_mask.sum() >= 2


def _check_line_stack_limit(skaters: pd.DataFrame) -> bool:
    """Allow up to 3 skaters from the same even-strength line (if info exists)."""
    col = _get_line_col(skaters)
    if not col:
        return True

    from collections import Counter

    lines = skaters[col].astype(str)
    cnt = Counter(lines)
    return all(v <= 3 for v in cnt.values())


# ---------------------------------------------------------------------
# Correlation + matchup scoring
# ---------------------------------------------------------------------
def _team_matchup_factor(team_rows: pd.DataFrame) -> float:
    """Use average PROJ as a proxy for matchup / game environment."""
    avg_proj = float(team_rows["PROJ"].mean())
    if not np.isfinite(avg_proj) or avg_proj <= 0:
        return 1.0

    # Baseline around 12 FPTS; clip to avoid crazy weights
    factor = avg_proj / 12.0
    factor = max(0.5, min(factor, 2.0))
    return factor


def _lineup_corr_score(skaters: pd.DataFrame) -> float:
    """Correlation score: team stacks, line stacks, PP1 stacks, matchup-weighted."""
    from collections import Counter

    score = 0.0

    # Team stacks with matchup weighting
    team_cnt = Counter(skaters["TEAM"])
    for team, n in team_cnt.items():
        if n < 2:
            continue
        team_rows = skaters[skaters["TEAM"] == team]
        matchup_factor = _team_matchup_factor(team_rows)
        score += (n - 1) * 1.5 * matchup_factor

    # Line stacks
    line_col = _get_line_col(skaters)
    if line_col:
        line_cnt = Counter(skaters[line_col].astype(str))
        for line_val, n in line_cnt.items():
            if n >= 2:
                line_rows = skaters[skaters[line_col].astype(str) == line_val]
                # DFS Line Strength System
                if "line_matchup_strength" in line_rows.columns:
                    lms = float(line_rows["line_matchup_strength"].mean())
                    lms = max(0.0, lms) / 100.0
                else:
                    lms = _team_matchup_factor(line_rows)
                score += (n - 1) * 2.0 * lms

    # PP1 stacks
    pp_col = _get_pp_col(skaters)
    if pp_col:
        vals = skaters[pp_col].astype(str).str.upper()
        pp1_rows = skaters[vals.isin(["1", "PP1"])]
        pp1_cnt = len(pp1_rows)
        if pp1_cnt >= 2:
            pp_factor = _team_matchup_factor(pp1_rows)
            score += (pp1_cnt - 1) * 2.0 * pp_factor

    return float(score)

def _lineup_goal_bonus(skaters: pd.DataFrame, center: float = 0.35, boost: float = 8.0) -> float:
    """
    Small additive bonus tied to the best (TEAM, LINE) stack's p_goal_line in this lineup.
    If p_goal_line is missing, returns 0.0.
    """
    if "p_goal_line" not in skaters.columns:
        return 0.0
    try:
        tmp = skaters.copy()
        # Normalize key columns defensively
        if "LINE" not in tmp.columns:
            if "line_tag" in tmp.columns:
                tmp["LINE"] = tmp["line_tag"].astype(str).str.upper().str.strip()
            elif "line_num" in tmp.columns:
                tmp["LINE"] = tmp["line_num"].apply(lambda n: f"FL{int(n)}" if pd.notna(n) else "FL?")
            else:
                tmp["LINE"] = "FL?"
        if "TEAM" not in tmp.columns and "team" in tmp.columns:
            tmp["TEAM"] = tmp["team"]
        tmp["p_goal_line"] = pd.to_numeric(tmp["p_goal_line"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        grp = tmp.groupby(["TEAM","LINE"])["p_goal_line"].mean().reset_index()
        if grp.empty:
            return 0.0
        best = float(grp["p_goal_line"].max())
        centered = max(0.0, best - float(center))
        return float(boost) * centered
    except Exception:
        return 0.0


# ---------------------------------------------------------------------
# Slate simulation helpers
# ---------------------------------------------------------------------
def _simulate_lineup(players: List[Dict[str, object]], n_sims: int = N_SIMS) -> Dict[str, float]:
    """Monte Carlo slate simulation for a single lineup with stack-aware correlation.

    This preserves the old interface/outputs but upgrades the engine to:
      - Use player-level volatility (FPTS_SIGMA when available).
      - Apply realistic positive correlation for linemates, same-team skaters, and PP1 stacks.
      - Apply negative correlation between goalies and their skaters / opposing skaters.
      - Simulate the full 9-player lineup via a multivariate normal draw.
    """
    if n_sims <= 0 or not players:
        return {
            "SIM_MEAN": 0.0,
            "SIM_STD": 0.0,
            "SIM_P75": 0.0,
            "SIM_P90": 0.0,
            "SIM_P95": 0.0,
            "BOOM_PCT": 0.0,
        }

    n = len(players)
    means = np.array([float(p.get("PROJ", 0.0)) for p in players], dtype=float)
    poss = [str(p.get("POS", "")).upper() for p in players]

    # -----------------------------
    # Per-player volatility
    # -----------------------------
    sigmas = []
    for p, pos, mu in zip(players, poss, means):
        base = max(mu, 1.0)
        vol_mult = VOL_FACTORS.get(pos, 0.9)

        if "FPTS_SIGMA" in p and p["FPTS_SIGMA"] is not None:
            try:
                sigma = float(p["FPTS_SIGMA"]) * vol_mult
            except (TypeError, ValueError):
                sigma = float(np.sqrt(base)) * vol_mult
        else:
            sigma = float(np.sqrt(base)) * vol_mult

        # avoid degenerate or insane sigmas
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.sqrt(base)) * vol_mult
        sigma = float(np.clip(sigma, 0.5, base * 2.0))
        sigmas.append(sigma)

    sigmas = np.array(sigmas, dtype=float)

    # -----------------------------
    # Stack-aware correlation matrix
    # -----------------------------
    corr = np.eye(n, dtype=float)

    # Pre-normalize some string fields for speed
    teams = [str(p.get("TEAM", "")).upper() for p in players]
    opps = [str(p.get("OPP", "")).upper() for p in players]
    lines = [str(p.get("LINE", "")).upper() for p in players]
    pp_lines = [str(p.get("PP_LINE", "")).upper() for p in players]

    for i in range(n):
        pos_i = poss[i]
        team_i = teams[i]
        opp_i = opps[i]
        line_i = lines[i]
        pp_i = pp_lines[i]

        for j in range(i + 1, n):
            pos_j = poss[j]
            team_j = teams[j]
            opp_j = opps[j]
            line_j = lines[j]
            pp_j = pp_lines[j]

            c = 0.0

            # -----------------------------
            # Positive correlation: skater  skater
            # -----------------------------
            if pos_i != "G" and pos_j != "G":
                # Same team core correlation
                if team_i and team_i == team_j:
                    c += 0.18

                    # Same even-strength line (L1/L2/L3)
                    if line_i and line_i == line_j:
                        c += 0.22

                    # Same PP1 unit
                    if (pp_i in ("PP1", "1")) and (pp_j in ("PP1", "1")):
                        c += 0.20

                # Same game, opposite teams - mild game-environment correlation
                if (team_i == opp_j) or (team_j == opp_i):
                    c += 0.05

            # -----------------------------
            # Negative correlation: goalie  skaters
            # -----------------------------
            if pos_i == "G" and pos_j != "G":
                # Own skaters vs own goalie
                if team_i and team_i == team_j:
                    c += -0.35
                # Skater vs opposing goalie
                if opp_i and opp_i == team_j:
                    c += -0.15

            if pos_j == "G" and pos_i != "G":
                if team_j and team_j == team_i:
                    c += -0.35
                if opp_j and opp_j == team_i:
                    c += -0.15

            # Clamp correlation
            c = float(np.clip(c, -0.50, 0.85))
            corr[i, j] = c
            corr[j, i] = c

    # Ensure perfect self-correlation
    np.fill_diagonal(corr, 1.0)

    # -----------------------------
    # Covariance and simulation
    # -----------------------------
    cov = corr * np.outer(sigmas, sigmas)

    # Small jitter on diagonal to reduce PSD issues
    jitter = 1e-6
    cov.flat[:: n + 1] += jitter

    try:
        draws = np.random.multivariate_normal(mean=means, cov=cov, size=n_sims)
        draws = np.clip(draws, 0.0, None)
        scores = draws.sum(axis=1)
    except np.linalg.LinAlgError:
        # Fallback to independent normal draws
        scores = np.zeros(n_sims, dtype=float)
        for mu, sigma in zip(means, sigmas):
            player_draws = np.random.normal(loc=mu, scale=sigma, size=n_sims)
            scores += np.clip(player_draws, 0.0, None)

    # -----------------------------
    # Summary stats
    # -----------------------------
    sim_mean = float(scores.mean())
    sim_std = float(scores.std(ddof=0))
    p75 = float(np.percentile(scores, 75))
    p90 = float(np.percentile(scores, 90))
    p95 = float(np.percentile(scores, 95))

    # "Boom" = lineup beats 110% of its projected total
    boom_threshold = 1.10 * float(means.sum())
    boom_pct = float((scores >= boom_threshold).mean())

    return {
        "SIM_MEAN": sim_mean,
        "SIM_STD": sim_std,
        "SIM_P75": p75,
        "SIM_P90": p90,
        "SIM_P95": p95,
        "BOOM_PCT": boom_pct,
    }


# ---------------------------------------------------------------------
# Build lineup row
# ---------------------------------------------------------------------
def _build_lineup_row(players_df: pd.DataFrame, corr_weight: float) -> Dict[str, object]:
    """Convert 9-player DataFrame into export row + corr metrics."""
    row: Dict[str, object] = {}

    total_sal = int(players_df["SAL"].sum())
    total_proj = float(players_df["PROJ"].sum())

    centers = players_df[players_df["POS"] == "C"].sort_values("PROJ", ascending=False)
    wings = players_df[players_df["POS"] == "W"].sort_values("PROJ", ascending=False)
    defense = players_df[players_df["POS"] == "D"].sort_values("PROJ", ascending=False)
    goalies = players_df[players_df["POS"] == "G"].sort_values("PROJ", ascending=False)

    if len(goalies) != 1:
        return {}

    # Slots
    for i, p in enumerate(centers.itertuples(index=False), start=1):
        row[f"C{i}_PLAYER"] = p.PLAYER
        row[f"C{i}_TEAM"] = p.TEAM
        row[f"C{i}_OPP"] = p.OPP
        row[f"C{i}_SAL"] = int(p.SAL)
        row[f"C{i}_PROJ"] = float(p.PROJ)

    for i, p in enumerate(wings.itertuples(index=False), start=1):
        row[f"W{i}_PLAYER"] = p.PLAYER
        row[f"W{i}_TEAM"] = p.TEAM
        row[f"W{i}_OPP"] = p.OPP
        row[f"W{i}_SAL"] = int(p.SAL)
        row[f"W{i}_PROJ"] = float(p.PROJ)

    for i, p in enumerate(defense.itertuples(index=False), start=1):
        row[f"D{i}_PLAYER"] = p.PLAYER
        row[f"D{i}_TEAM"] = p.TEAM
        row[f"D{i}_OPP"] = p.OPP
        row[f"D{i}_SAL"] = int(p.SAL)
        row[f"D{i}_PROJ"] = float(p.PROJ)

    g = goalies.iloc[0]
    row["G_PLAYER"] = g.PLAYER
    row["G_TEAM"] = g.TEAM
    row["G_OPP"] = g.OPP
    row["G_SAL"] = int(g.SAL)
    row["G_PROJ"] = float(g.PROJ)

    row["TOTAL_SAL"] = total_sal
    row["TOTAL_PROJ"] = total_proj

    # Correlation scoring
    skaters = players_df[players_df["POS"] != "G"].copy()
    corr_score = _lineup_corr_score(skaters)

    # NEW: Stack EV calculation
    stack_ev = _compute_stack_ev(skaters)
    row["STACK_EV"] = float(stack_ev)

    # Blend stack EV into score (modest weight)
    smart_score = (
        total_proj
        + corr_weight * corr_score
        + _lineup_goal_bonus(skaters)
        + stack_ev * 15.0
    )

    row["CORR_SCORE"] = float(corr_score)
    row["SMART_SCORE"] = float(smart_score)


    # Full player list for sim
    row["LINEUP_PLAYERS"] = []
    for p in players_df.itertuples(index=False):
        entry: Dict[str, object] = {
            "PLAYER": p.PLAYER,
            "POS": p.POS,
            "PROJ": float(p.PROJ),
        }
        # Extra context for stack-aware simulation
        if hasattr(p, "TEAM"):
            entry["TEAM"] = p.TEAM
        if hasattr(p, "OPP"):
            entry["OPP"] = p.OPP
        if hasattr(p, "LINE"):
            entry["LINE"] = getattr(p, "LINE")
        # Rotowire "PP LINE" column becomes PP_LINE attribute under itertuples
        if hasattr(p, "PP_LINE"):
            entry["PP_LINE"] = getattr(p, "PP_LINE")
        if hasattr(p, "fpts_sigma"):
            entry["FPTS_SIGMA"] = float(p.fpts_sigma)

        row["LINEUP_PLAYERS"].append(entry)

    return row

# ---------------------------------------------------------------------
# Projection-weighted sampling
# ---------------------------------------------------------------------
def _weighted_sample(n_players: int, k: int, proj_values: np.ndarray) -> np.ndarray:
    """Sample k distinct indices with probability  projection."""
    if n_players < k:
        raise ValueError("Not enough players to sample from.")

    w = np.maximum(proj_values, 0.0).astype(float)
    if w.sum() <= 0:
        return np.random.choice(n_players, size=k, replace=False)

    p = w / w.sum()
    return np.random.choice(n_players, size=k, replace=False, p=p)


# ---------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------
def generate_lineups(
    proj_file: Path = INPUT_PROJ_FILE,
    output_file: Path = OUTPUT_LINEUPS_FILE,
    iterations: int = ITERATIONS,
    num_lineups: int = NUM_LINEUPS,
    salary_cap: int = SALARY_CAP,
    min_salary: Optional[int] = 54300,
    corr_weight: float = CORR_WEIGHT,
    n_sims: int = N_SIMS,
) -> None:
    print(f" Loading FanDuel projections -> {proj_file}")
    df = load_fd_projections(proj_file)
        # -----------------------------------------
    # VALUE LOGIC (NEW)
    # -----------------------------------------
    # Salary per 1k
    df["SAL_K"] = df["SAL"] / 1000

    # Base value score = projection per $1K salary
    df["VALUE_SCORE"] = df["PROJ"] / df["SAL_K"].replace(0, np.nan)
    df["VALUE_SCORE"] = df["VALUE_SCORE"].fillna(0.0)

    # Build SAMPLE_SCORE (used for player sampling)
    df["SAMPLE_SCORE"] = (
        0.65 * df["PROJ"] +
        0.35 * df["VALUE_SCORE"]
    )

    # Cheap PP1 boost
    if "PP_LINE" in df.columns:
        df.loc[(df["PP_LINE"].astype(str).isin(["1","PP1"])) & (df["SAL"] <= 4500),
               "SAMPLE_SCORE"] *= 1.12

    # Cheap L1 boost
    if "LINE" in df.columns:
        df.loc[(df["LINE"].astype(str).isin(["1","L1"])) & (df["SAL"] <= 4200),
               "SAMPLE_SCORE"] *= 1.08

    # Clip to prevent extreme values
    df["SAMPLE_SCORE"] = df["SAMPLE_SCORE"].clip(lower=0.01)
    print(f"   Loaded {len(df)} players with valid projections.")

    pool_C = df[df["POS"] == "C"].reset_index(drop=True)
    pool_W = df[df["POS"] == "W"].reset_index(drop=True)
    pool_D = df[df["POS"] == "D"].reset_index(drop=True)
    pool_G = df[df["POS"] == "G"].reset_index(drop=True)

    if pool_C.empty or pool_W.empty or pool_D.empty or pool_G.empty:
        raise ValueError("One of the POS pools (C/W/D/G) is empty after filtering.")

    # random.seed(RNG_SEED)
    # np.random.seed(RNG_SEED)

    best_lineups: List[Dict[str, object]] = []

    print(
        f" Generating up to {num_lineups} lineups with {iterations} attempts "
        f"(salary_cap={salary_cap}, min_salary={min_salary}, corr_weight={corr_weight})..."
    )

    for _ in range(iterations):
        if len(pool_C) < 2 or len(pool_W) < 4 or len(pool_D) < 2 or len(pool_G) < 1:
            raise ValueError("Not enough players in at least one positional pool.")

        # Projection-weighted sampling for each position
        # VALUE-AWARE sampling (uses SAMPLE_SCORE)
        c_idx = _weighted_sample(len(pool_C), 2, pool_C["SAMPLE_SCORE"].values)
        w_idx = _weighted_sample(len(pool_W), 4, pool_W["SAMPLE_SCORE"].values)
        d_idx = _weighted_sample(len(pool_D), 2, pool_D["SAMPLE_SCORE"].values)
        g_idx = _weighted_sample(len(pool_G), 1, pool_G["SAMPLE_SCORE"].values)[0]


        centers = pool_C.iloc[c_idx]
        wings = pool_W.iloc[w_idx]
        defense = pool_D.iloc[d_idx]
        goalie = pool_G.iloc[[g_idx]]

        players_df = pd.concat([centers, wings, defense, goalie], ignore_index=True)

        # No duplicates
        if players_df["PLAYER"].duplicated().any():
            continue

        total_sal = int(players_df["SAL"].sum())
        if min_salary is not None and total_sal < min_salary:
            continue
        if total_sal > salary_cap:
            continue

        # Stack + goalie rules
        skaters = players_df[players_df["POS"] != "G"].copy()

        if not _check_team_stack_rules(skaters["TEAM"].tolist()):
            continue
        if not _check_pp1_stack(skaters):
            continue
        if not _check_line_stack_limit(skaters):
            continue

        g_team = goalie.iloc[0]["TEAM"]
        g_opp = goalie.iloc[0]["OPP"]
        if not _check_goalie_rules(g_team, g_opp, skaters):
            continue

        row = _build_lineup_row(players_df, corr_weight=corr_weight)
        if not row:
            continue

        best_lineups.append(row)

        # Manage buffer using SMART_SCORE (no sim stats yet)
        if len(best_lineups) >= num_lineups * 4:
            best_lineups = sorted(
                best_lineups,
                key=lambda x: x["SMART_SCORE"],
                reverse=True,
            )[: num_lineups * 2]

    if not best_lineups:
        raise RuntimeError("No valid lineups generated under current constraints.")

    # First: run slate simulation on all candidate lineups
    print(f" Running slate simulation for {len(best_lineups)} lineups (n_sims={n_sims})...")
    for row in best_lineups:
        players = row.get("LINEUP_PLAYERS", [])
        sim_stats = _simulate_lineup(players, n_sims=n_sims)
        row.update(sim_stats)
        # Remove heavy internal list before writing
        row.pop("LINEUP_PLAYERS", None)

    # Now: final sort by sim-based score and trim to num_lineups
    best_lineups = sorted(
        best_lineups,
        key=lambda x: x.get("SIM_MEAN", 0.0) + 20.0 * x.get("BOOM_PCT", 0.0),
        reverse=True,
    )[:num_lineups]

    df_out = pd.DataFrame(best_lineups)

    print(f" Writing {len(df_out)} lineups -> {output_file}")
    df_out.to_csv(output_file, index=False)
    print(" Done.")

if __name__ == "__main__":
    generate_lineups()
