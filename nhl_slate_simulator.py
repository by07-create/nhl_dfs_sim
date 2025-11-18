# nhl_slate_simulator.py
#
# Full-slate Monte Carlo simulator for NHL FD lineups
# - Uses nhl_fd_projections.csv (mean + sigma + matchup features)
# - Uses a lineup CSV (~500 lineups) and simulates 100k slates
# - Outputs: nhl_lineup_sim_leaderboard.csv (full leaderboard)
#
# This DOES NOT touch or modify any of your existing files.
# It only reads projections + lineups and writes its own CSV.

import numpy as np
import pandas as pd
from pathlib import Path
import math

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
APP_ROOT = Path.home() / "OneDrive" / "Documents" / "Bo - Python Apps" / "NHL Simulator"

PROJ_FILE   = APP_ROOT / "nhl_fd_projections.csv"
LINEUP_FILE = APP_ROOT / "nhl_fd_lineups.csv"   # adjust if your builder uses a different name

OUTPUT_FILE = APP_ROOT / "nhl_lineup_sim_leaderboard.csv"

MAX_LINEUPS = 500
N_SIMS      = 100_000
BATCH_SIZE  = 1000       # sims per batch (keeps memory reasonable)
RNG_SEED    = 42

# Volatility of environment factors
GAME_SIGMA      = 0.12    # game-level pace/total volatility
TEAM_OFF_SIGMA  = 0.10    # team offensive environment
LINE_SIGMA      = 0.15    # line "hot/cold"
GOALIE_ANTI_EXP = -0.30   # exponent for goalie vs opp-offense anti-corr


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _normalize_name(s: str) -> str:
    return str(s).strip().lower()


def load_projections(path: Path) -> pd.DataFrame:
    print(f" Loading projections -> {path}")
    df = pd.read_csv(path)

    required_cols = ["PLAYER", "TEAM", "OPP", "FPTS", "fpts_sigma"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Projections file missing required columns: {missing}")

    if "is_goalie" not in df.columns:
        # Fallback: treat POS == 'G' as goalie
        df["is_goalie"] = df["POS"].astype(str).str.upper().eq("G")

    # Normalize names for joins
    df["_norm_name"] = df["PLAYER"].map(_normalize_name)

    # Optional matchup / line columns
    if "matchup_mult" not in df.columns:
        df["matchup_mult"] = 1.0
    df["matchup_mult"] = df["matchup_mult"].fillna(1.0).astype(float)

    # Line key - try env_key, fall back to TEAM+line_num, else per-player
    if "env_key" in df.columns:
        line_key = df["env_key"].fillna(
            df["TEAM"].astype(str).str.upper() + "_L0"
        )
    elif "line_num" in df.columns:
        line_key = (
            df["TEAM"].astype(str).str.upper()
            + "_L"
            + df["line_num"].fillna(0).astype(int).astype(str)
        )
    else:
        line_key = df["TEAM"].astype(str).str.upper() + "_" + df["_norm_name"]

    df["_line_key"] = line_key.astype(str)

    # Game key: team + opp sorted so both sides share the same game id
    t = df["TEAM"].astype(str).str.upper()
    o = df["OPP"].astype(str).str.upper()
    game_key = np.where(
        t < o,
        t + "@" + o,
        o + "@" + t,
    )
    df["_game_key"] = game_key

    return df


def load_lineups(path, df_players):
    """Load FD lineups from CSV. Supports:
       - LINEUP column (comma-separated names)
       - FD slot columns C1..G
       - PLAYER1..PLAYER9
       - Slot detail format (C1_PLAYER, W2_PLAYER, etc.)  <-- YOUR FORMAT
    """

    df_lu = pd.read_csv(path)
    df_lu.columns = [str(c).strip().upper() for c in df_lu.columns]

    # ------------------------------
    # CASE 1 - Already has LINEUP
    # ------------------------------
    if "LINEUP" in df_lu.columns:
        return df_lu

    # ------------------------------
    # CASE 2 - FD SLOT (C1..G)
    # ------------------------------
    fd_slots = ["C1", "C2", "W1", "W2", "W3", "W4", "D1", "D2", "G"]
    if all(col in df_lu.columns for col in fd_slots):
        df_lu["LINEUP"] = df_lu[fd_slots].astype(str).agg(",".join, axis=1)
        return df_lu

    # ---------------------------------------------------------
    # CASE 3 - PLAYER1..PLAYER9 columns
    # ---------------------------------------------------------
    player_cols = [c for c in df_lu.columns if c.startswith("PLAYER")]
    player_cols = sorted(player_cols, key=lambda x: int(x.replace("PLAYER", "")) if x.replace("PLAYER","").isdigit() else 999)

    if len(player_cols) == 9:
        print(" Detected PLAYER1..PLAYER9 format - building LINEUP column.")
        df_lu["LINEUP"] = df_lu[player_cols].astype(str).agg(",".join, axis=1)
        return df_lu

    # ---------------------------------------------------------
    # CASE 4 - YOUR FORMAT (C1_PLAYER, W3_PLAYER, D2_PLAYER...)
    # ---------------------------------------------------------
    slot_player_cols = [
        "C1_PLAYER","C2_PLAYER",
        "W1_PLAYER","W2_PLAYER","W3_PLAYER","W4_PLAYER",
        "D1_PLAYER","D2_PLAYER",
        "G_PLAYER"
    ]

    if all(col in df_lu.columns for col in slot_player_cols):
        print(" Detected detailed slot format - extracting *_PLAYER columns.")
        df_lu["LINEUP"] = df_lu[slot_player_cols].astype(str).agg(",".join, axis=1)

        # -----------------------------------------------------
        #  Build _player_indices for simulator
        # -----------------------------------------------------
        name_map = df_players.set_index("_norm_name").index

        player_index_lookup = {
            n: i
            for i, n in enumerate(df_players["_norm_name"].tolist())
        }

        indices = []
        for lineup_str in df_lu["LINEUP"]:
            names = [s.strip().lower() for s in lineup_str.split(",")]
            idxs = []
            for nm in names:
                if nm in player_index_lookup:
                    idxs.append(player_index_lookup[nm])
                else:
                    print(f" WARNING: Could not match player '{nm}' in projections.")
            indices.append(idxs)

        df_lu["_player_indices"] = indices

        return df_lu


    # ---------------------------------------------------------
    # NOTHING MATCHED -> Error
    # ---------------------------------------------------------
    raise ValueError(
        f"Lineups file not recognized. Columns found: {df_lu.columns.tolist()}"
    )

def build_lookup_ids(df: pd.DataFrame):
    # Teams
    teams = sorted(df["TEAM"].astype(str).str.upper().unique().tolist())
    team_to_id = {t: i for i, t in enumerate(teams)}
    team_id = df["TEAM"].astype(str).str.upper().map(team_to_id).to_numpy()

    # Opp teams
    opp_team_id = df["OPP"].astype(str).str.upper().map(team_to_id).to_numpy()

    # Games
    games = sorted(df["_game_key"].unique().tolist())
    game_to_id = {g: i for i, g in enumerate(games)}
    game_id = df["_game_key"].map(game_to_id).to_numpy()

    # Lines
    lines = sorted(df["_line_key"].astype(str).unique().tolist())
    line_to_id = {lk: i for i, lk in enumerate(lines)}
    line_id = df["_line_key"].astype(str).map(line_to_id).to_numpy()

    return {
        "team_to_id": team_to_id,
        "team_id": team_id.astype(np.int32),
        "opp_team_id": opp_team_id.astype(np.int32),
        "game_to_id": game_to_id,
        "game_id": game_id.astype(np.int32),
        "line_to_id": line_to_id,
        "line_id": line_id.astype(np.int32),
    }


def build_lineup_matrix(df_players: pd.DataFrame, df_lineups: pd.DataFrame) -> np.ndarray:
    n_players = len(df_players)
    n_lineups = len(df_lineups)

    mat = np.zeros((n_lineups, n_players), dtype=np.float32)

    for li, idxs in enumerate(df_lineups["_player_indices"]):
        mat[li, idxs] = 1.0

    return mat

# ------------------------------
# Stack EV in slate simulator
# ------------------------------
STACK_LINE_SHARES = {
    "1": 0.40, "2": 0.30, "3": 0.20, "4": 0.10,
    "L1": 0.40, "L2": 0.30, "L3": 0.20, "L4": 0.10,
}

def _estimate_team_ev_from_proj(df_team: pd.DataFrame) -> float:
    avg = float(df_team["FPTS"].mean())
    if not np.isfinite(avg) or avg <= 0:
        return 1.8
    return float(avg / 12.0)

def _estimate_line_ev(team_ev: float, line_val: str) -> float:
    s = STACK_LINE_SHARES.get(str(line_val).upper(), 0.10)
    return float(team_ev * s)

def _compute_stack_ev_for_lineup(players: pd.DataFrame) -> float:
    sk = players[players["POS"] != "G"].copy()
    if sk.empty:
        return 0.0

    if "_line_key" not in sk.columns and "LINE" not in sk.columns:
        # fallback: team-only
        team_evs = []
        for team in sk["TEAM"].unique():
            team_rows = sk[sk["TEAM"] == team]
            team_evs.append(_estimate_team_ev_from_proj(team_rows))
        return max(team_evs) if team_evs else 0.0

    # Determine line col
    if "LINE" in sk.columns:
        line_col = "LINE"
    elif "_line_key" in sk.columns:
        line_col = "_line_key"
    else:
        line_col = "TEAM"

    # Team EV per team
    team_ev_map = {}
    for team in sk["TEAM"].unique():
        team_ev_map[team] = _estimate_team_ev_from_proj(sk[sk["TEAM"] == team])

    best_line_ev = 0.0
    for (team, line), grp in sk.groupby(["TEAM", line_col]):
        team_ev = team_ev_map.get(team, 1.8)
        line_ev = _estimate_line_ev(team_ev, line)
        best_line_ev = max(best_line_ev, line_ev)

    x = (best_line_ev - 0.65) * 4.0
    boom = 1 / (1 + np.exp(-x))
    return float(boom)

# -------------------------------------------------------
# Monte Carlo Slate Simulator
# -------------------------------------------------------

def simulate_slate(df_players: pd.DataFrame, df_lineups: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)

    lookups = build_lookup_ids(df_players)

    team_id      = lookups["team_id"]
    opp_team_id  = lookups["opp_team_id"]
    game_id      = lookups["game_id"]
    line_id      = lookups["line_id"]

    n_players = len(df_players)
    n_lineups = len(df_lineups)

    n_teams = len(lookups["team_to_id"])
    n_games = len(lookups["game_to_id"])
    n_lines = len(lookups["line_to_id"])

    print(f" Players: {n_players},  Lineups: {n_lineups}")
    print(f" Teams: {n_teams}, Games: {n_games}, Lines: {n_lines}")

    mu = df_players["FPTS"].to_numpy(dtype=np.float32)
    sigma = df_players["fpts_sigma"].to_numpy(dtype=np.float32)
    matchup_mult = df_players["matchup_mult"].to_numpy(dtype=np.float32)

    is_goalie = df_players["is_goalie"].astype(bool).to_numpy()
    goalie_idx = np.where(is_goalie)[0]
    has_goalies = len(goalie_idx) > 0

    lineup_mat = build_lineup_matrix(df_players, df_lineups)  # (L, P)

    # Storage for lineup scores per sim (float32 to keep memory down)
    scores = np.zeros((n_lineups, N_SIMS), dtype=np.float32)

    sims_done = 0
    batch_index = 0

    while sims_done < N_SIMS:
        b = min(BATCH_SIZE, N_SIMS - sims_done)
        batch_index += 1
        print(f" Sim batch {batch_index}: {sims_done} -> {sims_done + b}")

        # Environment draws
        # Game-level pace/total
        game_factor = rng.lognormal(mean=0.0, sigma=GAME_SIGMA, size=(b, n_games)).astype(np.float32)
        # Team offensive environment
        team_off_factor = rng.lognormal(mean=0.0, sigma=TEAM_OFF_SIGMA, size=(b, n_teams)).astype(np.float32)
        # Line "hot/cold"
        line_factor = rng.lognormal(mean=0.0, sigma=LINE_SIGMA, size=(b, n_lines)).astype(np.float32)

        # Broadcast to players
        # Each is shape (b, P)
        gf = game_factor[:, game_id]
        tf_off = team_off_factor[:, team_id]
        lf = line_factor[:, line_id]

        mult = gf * tf_off * lf * matchup_mult[None, :]

        # Goalie vs offense anti-correlation:
        if has_goalies:
            # Opponent attack factor for each goalie
            opp_tf = team_off_factor[:, opp_team_id[goalie_idx]]  # (b, n_goalies)
            goalie_adjust = np.power(opp_tf, GOALIE_ANTI_EXP, dtype=np.float32)
            mult[:, goalie_idx] *= goalie_adjust

        # Effective mean per sim & player
        mu_eff = mu[None, :] * mult

        # Sample player outcomes
        eps = rng.standard_normal(size=(b, n_players)).astype(np.float32)
        samples = mu_eff + sigma[None, :] * eps
        samples = np.clip(samples, 0.0, None)  # no negative fantasy scores

        # Lineup scores: (b x P) @ (P x L) -> (b x L)
        batch_scores = samples @ lineup_mat.T
        scores[:, sims_done : sims_done + b] = batch_scores.T

        sims_done += b

    print(" Simulation complete. Computing summary stats...")

    # ---------------------------------------------------
    # Summaries
    # ---------------------------------------------------
    mean_scores = scores.mean(axis=1)
    std_scores = scores.std(axis=1)

    p50 = np.percentile(scores, 50, axis=1)
    p75 = np.percentile(scores, 75, axis=1)
    p90 = np.percentile(scores, 90, axis=1)
    p95 = np.percentile(scores, 95, axis=1)
    p99 = np.percentile(scores, 99, axis=1)

    # Top 1% finish rate (approx): fraction of sims where lineup is in top 1% of scores
    # For each sim (column), find cutoff for top 1%
    # Handle small lineup counts safely
    if n_lineups <= 1:
        k = 0
    else:
        k = int(math.ceil(n_lineups * 0.99)) - 1
        k = max(0, min(k, n_lineups - 1))
    sorted_scores = np.sort(scores, axis=0)  # sort each column
    thresh = sorted_scores[k, :]            # shape (N_SIMS,)

    win_mask = scores >= thresh[None, :]
    win_rate = win_mask.mean(axis=1)

    # Build leaderboard DataFrame
    out = df_lineups.copy()
    out["sim_mean"] = mean_scores
    out["sim_std"] = std_scores
    out["p50"] = p50
    out["p75"] = p75
    out["p90"] = p90
    out["p95"] = p95
    out["p99"] = p99
    out["top1_pct"] = win_rate

    # ------------------------------------------
    # Add Stack EV to slate simulation leaderboard
    # ------------------------------------------
    stack_evs = []
    for _, row in df_lineups.iterrows():
        names = [x.strip() for x in row["LINEUP"].split(",")]
        sub = df_players[df_players["PLAYER"].str.lower().isin([n.lower() for n in names])]
        sev = _compute_stack_ev_for_lineup(sub)
        stack_evs.append(sev)

    out["stack_ev"] = stack_evs

    # Update final EV score using StackEV
    out["ev_score"] = (
        out["sim_mean"]
        + 2.5 * out["top1_pct"] * out["p95"]
        + out["stack_ev"] * 15.0
    )


    # Sort best -> worst by EV score
    out = out.sort_values("ev_score", ascending=False).reset_index(drop=True)

    return out


def main():
    df_players = load_projections(PROJ_FILE)
    df_lineups = load_lineups(LINEUP_FILE, df_players)

    leaderboard = simulate_slate(df_players, df_lineups)

    print(f" Writing leaderboard -> {OUTPUT_FILE}")
    leaderboard.to_csv(OUTPUT_FILE, index=False)
    print(" Done.")


if __name__ == "__main__":
    main()
