# merge_nhl_data.py  (CLOUD-SAFE PATH PATCH — NO LOGIC CHANGES)

import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata

# -----------------------------
# Paths  (PATCHED FOR STREAMLIT)
# -----------------------------
# Use the directory where this script lives
APP_ROOT = Path(__file__).parent.resolve()

# Streamlit uploader will save RW here
DATA_DIR = APP_ROOT

RW_FILE      = DATA_DIR / "rw-nhl-player-pool.xlsx"
PP_FILE      = DATA_DIR / "nhl-stats-power-play.xlsx"     # optional
OUTPUT_FILE  = APP_ROOT / "merged_nhl_player_pool.csv"


# -----------------------------
# MoneyPuck HTTP config
# -----------------------------
SEASON = 2025
GAME_TYPE = "regular"

MP_BASE = "https://moneypuck.com/moneypuck"

# Skaters
MP_SKATERS_SEASON = f"{MP_BASE}/playerData/seasonSummary/{SEASON}/{GAME_TYPE}/skaters.csv"
MP_SKATERS_L10    = f"{MP_BASE}/playerData/seasonSummary/{SEASON}/{GAME_TYPE}/skaters_10.csv"
MP_SKATERS_L20    = f"{MP_BASE}/playerData/seasonSummary/{SEASON}/{GAME_TYPE}/skaters_20.csv"

# Goalies
MP_GOALIES_SEASON = f"{MP_BASE}/playerData/seasonSummary/{SEASON}/{GAME_TYPE}/goalies.csv"
MP_GOALIES_L10    = f"{MP_BASE}/playerData/seasonSummary/{SEASON}/{GAME_TYPE}/goalies_10.csv"
MP_GOALIES_L20    = f"{MP_BASE}/playerData/seasonSummary/{SEASON}/{GAME_TYPE}/goalies_20.csv"

# Teams - only L10/L20 here; no teams.csv
MP_TEAMS_L10      = f"{MP_BASE}/playerData/seasonSummary/{SEASON}/{GAME_TYPE}/teams_10.csv"
MP_TEAMS_L20      = f"{MP_BASE}/playerData/seasonSummary/{SEASON}/{GAME_TYPE}/teams_20.csv"

# Lines
MP_LINES          = f"{MP_BASE}/playerData/seasonSummary/{SEASON}/{GAME_TYPE}/lines.csv"


# -----------------------------
# Helpers
# -----------------------------

def _load_mp_csv(url: str, label: str) -> pd.DataFrame:
    print(f" Fetching MoneyPuck {label} -> {url}")
    try:
        df = pd.read_csv(url)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        print(f" Failed to load {label} from {url}: {e}")
        return pd.DataFrame()


def _blend_recency(base: pd.Series, l10: pd.Series, l20: pd.Series) -> pd.Series:
    base = pd.to_numeric(base, errors="coerce")
    l10 = pd.to_numeric(l10, errors="coerce")
    l20 = pd.to_numeric(l20, errors="coerce")
    return 0.20 * base + 0.50 * l10 + 0.30 * l20


def _per60(total: pd.Series, icetime_sec: pd.Series) -> pd.Series:
    total = pd.to_numeric(total, errors="coerce")
    icetime_sec = pd.to_numeric(icetime_sec, errors="coerce")
    hours = icetime_sec / 3600.0
    hours = hours.replace(0, np.nan)
    rate = total / hours
    return rate.replace([np.inf, -np.inf], np.nan)


NAME_FIX = {
    "mitchell marner": "mitch marner",
    "matt knies": "matthew knies",
    "zachary werenski": "zach werenski",
    "joshua morrissey": "josh morrissey",
    "tim stuetzle": "tim sttzle",
    "charles mcavoy": "charlie mcavoy",
    "dmitry voronkov": "dmitri voronkov",
    "alexander ovechkin": "alex ovechkin",
    "juraj slafkovsky": "juraj slafkovsk",
    "michael matheson": "mike matheson",
    "artyom zub": "artem zub",
    "matthew coronato": "matt coronato",
    "samuel bennett": "sam bennett",
    "martin fehervary": "martin fehrvry",
    "zachary bolduc": "zack bolduc",
    "anthony deangelo": "tony deangelo",
    "john-jason peterka": "jj peterka",
    "dmitry simashev": "dmitri simashev",
    "pierre-olivier joseph": "p.o. joseph",
    "olli maatta": "olli mtt",
    "yegor zamula": "egor zamula",
    "daniel vladar": "dan vladar",
    "alexis lafreniere": "alexis lafrenire",
    "axel sandin": "axel sandin-pellikka",
    "matthew savoie": "matt savoie",
    "michael anderson": "mikey anderson",
    "matthew grzelcyk": "matt grzelcyk",
    "luca del": "luca del bel belluz",
    "justin hrycowian": "justin hryckowian",
    "isac lundestrom": "isac lundestrm",
    "jeffrey truchon-viel": "jeffrey viel",
    "trevor van": "trevor van riemsdyk",
    "josh mahura": "joshua mahura",
    "phil myers": "philippe myers",
    "iaroslav askarov": "yaroslav askarov",
    "samuel montembeault": "sam montembeault",
    "leevi merilainen": "leevi merilinen",
    "cameron talbot": "cam talbot",
    "jon quick": "jonathan quick",
    "matthew murray": "matt murray",
    "matthew boldy": "matt boldy",
    "joel eriksson": "joel eriksson ek",
    "jacob middleton": "jake middleton",
    "janis moser": "j.j. moser",
    "vincent hinostroza": "vinnie hinostroza",
    "andrei vasilevskii": "andrei vasilevskiy",
    "zach aston-reese": "zachary aston-reese",
    "anthony-john greer": "a.j. greer",
    "evgeni dadonov": "evgenii dadonov",
    "maxim shabanov": "maxim tsyplakov",
    "james van": "james van riemsdyk",
    "samuel blais": "sammy blais",
    "joel daccord": "joey daccord",
}

# ---------------------------------------------------------
# Normalize NAME_FIX Keys + Values (critical fix)
# ---------------------------------------------------------
def _normalize_for_map(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    for ch in [".", ",", "'", "\"", "-", "’"]:
        s = s.replace(ch, "")
    # collapse spaces
    return " ".join(s.split())

# Create normalized name map
NAME_FIX = {
    _normalize_for_map(k): _normalize_for_map(v)
    for k, v in NAME_FIX.items()
}


def clean_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    s = (
        s.replace(".", "")
         .replace("'", "")
         .replace("-", " ")
         .replace("'", "")
         .strip()
    )
    s = " ".join(s.split())
    return NAME_FIX.get(s, s)

def normalize_team(s: str) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().upper()


def safe_int(x):
    try:
        return int(str(x).strip())
    except Exception:
        return None


# -----------------------------
# Rotowire
# -----------------------------

def load_rotowire(path: Path) -> pd.DataFrame:
    print(" Loading Rotowire slate...")
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    if "PLAYER" not in df.columns or "TEAM" not in df.columns:
        raise ValueError("Rotowire file is missing 'PLAYER' and/or 'TEAM' columns.")

    df["player_clean"] = df["PLAYER"].apply(clean_name)
    df["TEAM"] = df["TEAM"].apply(normalize_team)
    if "OPP" in df.columns:
        df["OPP"] = df["OPP"].apply(normalize_team)

    pos_map = {"C/UTIL": "C", "W/UTIL": "W", "D/UTIL": "D", "G": "G"}
    df["POS"] = df["POS"].astype(str).str.strip()
    df["base_pos"] = df["POS"].map(pos_map).fillna(df["POS"])

    if "LINE" in df.columns:
        df["line_num"] = df["LINE"].apply(safe_int)
    else:
        df["line_num"] = None

    if "PP LINE" in df.columns:
        df["pp_unit"] = df["PP LINE"].apply(safe_int)
    else:
        df["pp_unit"] = None

    df["env_key"] = None
    mask_fwd = df["base_pos"].isin(["C", "W"]) & df["line_num"].notna()
    df.loc[mask_fwd, "env_key"] = (
        df.loc[mask_fwd, "TEAM"] + "_L" + df.loc[mask_fwd, "line_num"].astype(int).astype(str)
    )
    mask_def = df["base_pos"].eq("D") & df["line_num"].notna()
    df.loc[mask_def, "env_key"] = (
        df.loc[mask_def, "TEAM"] + "_P" + df.loc[mask_def, "line_num"].astype(int).astype(str)
    )

    df["is_goalie"] = df["base_pos"].eq("G")
    return df


# -----------------------------
# Skaters
# -----------------------------

def load_skaters() -> pd.DataFrame:
    print(" Loading MoneyPuck skaters (season + L10 + L20)...")
    dfs = _load_mp_csv(MP_SKATERS_SEASON, "skaters seasonSummary")
    if dfs.empty:
        raise RuntimeError("MoneyPuck skaters seasonSummary is empty.")

    if not {"name", "team", "situation", "icetime"}.issubset(dfs.columns):
        raise ValueError("skaters.csv missing required columns: name, team, situation, icetime")

    dfs["team"] = dfs["team"].apply(normalize_team)
    dfs["player_clean"] = dfs["name"].apply(clean_name)

    df_all = dfs[dfs["situation"] == "all"].copy()
    if df_all.empty:
        df_all = dfs[dfs["situation"] == "5on5"].copy()

    df_all = df_all.sort_values(
        ["player_clean", "team", "icetime"], ascending=[True, True, False]
    ).drop_duplicates(subset=["player_clean", "team"], keep="first")

    base_cols = [c for c in df_all.columns if c != "situation"]
    base = df_all[base_cols].copy()

    df10 = _load_mp_csv(MP_SKATERS_L10, "skaters L10")
    if not df10.empty:
        df10["team"] = df10["team"].apply(normalize_team)
        df10["player_clean"] = df10["name"].apply(clean_name)
        s10 = df10[df10["situation"] == "all"].copy()
        if s10.empty:
            s10 = df10[df10["situation"] == "5on5"].copy()
        s10 = s10.sort_values(
            ["player_clean", "team", "icetime"], ascending=[True, True, False]
        ).drop_duplicates(subset=["player_clean", "team"], keep="first")
        s10 = s10[[c for c in s10.columns if c != "situation"]]
        s10 = s10.rename(columns={c: f"{c}_L10" for c in s10.columns if c not in {"player_clean", "team"}})
        base = base.merge(s10, how="left", on=["player_clean", "team"])

    df20 = _load_mp_csv(MP_SKATERS_L20, "skaters L20")
    if not df20.empty:
        df20["team"] = df20["team"].apply(normalize_team)
        df20["player_clean"] = df20["name"].apply(clean_name)
        s20 = df20[df20["situation"] == "all"].copy()
        if s20.empty:
            s20 = df20[df20["situation"] == "5on5"].copy()
        s20 = s20.sort_values(
            ["player_clean", "team", "icetime"], ascending=[True, True, False]
        ).drop_duplicates(subset=["player_clean", "team"], keep="first")
        s20 = s20[[c for c in s20.columns if c != "situation"]]
        s20 = s20.rename(columns={c: f"{c}_L20" for c in s20.columns if c not in {"player_clean", "team"}})
        base = base.merge(s20, how="left", on=["player_clean", "team"])

    for raw_col, out_col in [
        ("I_F_xGoals", "xG_per60_recency"),
        ("I_F_shotsOnGoal", "SOG_per60_recency"),
        ("I_F_shotAttempts", "CF_per60_recency"),
    ]:
        if raw_col not in base.columns:
            continue
        base_rate = _per60(base[raw_col], base.get("icetime"))
        l10_rate = _per60(
            base.get(f"{raw_col}_L10", np.nan),
            base.get("icetime_L10", base.get("icetime")),
        )
        l20_rate = _per60(
            base.get(f"{raw_col}_L20", np.nan),
            base.get("icetime_L20", base.get("icetime")),
        )
        rec = _blend_recency(base_rate, l10_rate, l20_rate)
        base[out_col] = rec.fillna(base_rate).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return base


# -----------------------------
# Teams - L10/L20 only, L20 as season proxy
# -----------------------------

def load_teams() -> pd.DataFrame:
    print(" Loading MoneyPuck teams (L10 + L20, L20 as season proxy)...")
    t10 = _load_mp_csv(MP_TEAMS_L10, "teams L10")
    t20 = _load_mp_csv(MP_TEAMS_L20, "teams L20")

    if t10.empty and t20.empty:
        print(" No team L10/L20 data available - team env will be neutral.")
        return pd.DataFrame()

    def prep(df: pd.DataFrame, label: str) -> pd.DataFrame:
        if df.empty:
            return df
        if "team" not in df.columns or "situation" not in df.columns:
            print(f" teams {label} missing 'team' or 'situation'; skipping.")
            return pd.DataFrame()
        df["team"] = df["team"].apply(normalize_team)
        d = df[df["situation"] == "all"].copy()
        if d.empty:
            d = df[df["situation"] == "5on5"].copy()
        if "iceTime" in d.columns:
            d = d.sort_values(["team", "iceTime"], ascending=[True, False])
        d = d.drop_duplicates(subset=["team"], keep="first")
        core = [
            "team",
            "games_played",
            "xGoalsFor",
            "xGoalsAgainst",
            "shotsOnGoalFor",
            "shotAttemptsFor",
            "shotsOnGoalAgainst",
            "shotAttemptsAgainst",
        ]
        core = [c for c in core if c in d.columns]
        return d[core]

    a10 = prep(t10, "L10")
    a20 = prep(t20, "L20")

    if a20.empty:
        base = a10.copy()
        base = base.rename(columns={c: f"{c}_L10" for c in base.columns if c != "team"})
        base["games_played"] = base.get("games_played_L10", np.nan)
        base["xGoalsFor"] = base.get("xGoalsFor_L10", np.nan)
        base["xGoalsAgainst"] = base.get("xGoalsAgainst_L10", np.nan)
        gp = pd.to_numeric(base["games_played"], errors="coerce").replace(0, np.nan)
        xgf_pg = pd.to_numeric(base["xGoalsFor"], errors="coerce") / gp
        xga_pg = pd.to_numeric(base["xGoalsAgainst"], errors="coerce") / gp
        base["xGF_pg_recency"] = xgf_pg
        base["xGA_pg_recency"] = xga_pg
        return base

    base = a20.copy()
    base = base.rename(columns={c: f"{c}_L20" for c in base.columns if c != "team"})
    if not a10.empty:
        a10 = a10.rename(columns={c: f"{c}_L10" for c in a10.columns if c != "team"})
        base = base.merge(a10, how="left", on="team")

    base["games_played"] = base.get("games_played_L20", np.nan)
    base["xGoalsFor"] = base.get("xGoalsFor_L20", np.nan)
    base["xGoalsAgainst"] = base.get("xGoalsAgainst_L20", np.nan)
    base["shotsOnGoalFor"] = base.get("shotsOnGoalFor_L20", np.nan)
    base["shotAttemptsFor"] = base.get("shotAttemptsFor_L20", np.nan)
    base["shotsOnGoalAgainst"] = base.get("shotsOnGoalAgainst_L20", np.nan)
    base["shotAttemptsAgainst"] = base.get("shotAttemptsAgainst_L20", np.nan)

    gp_season = pd.to_numeric(base["games_played"], errors="coerce").replace(0, np.nan)
    gp_10 = pd.to_numeric(base.get("games_played_L10", base["games_played"]), errors="coerce").replace(0, np.nan)
    gp_20 = pd.to_numeric(base.get("games_played_L20", base["games_played"]), errors="coerce").replace(0, np.nan)

    def recency_pg(stat_base: str, stat_l10: str, stat_l20: str, out_col: str):
        sb = pd.to_numeric(base.get(stat_base, np.nan), errors="coerce")
        s10 = pd.to_numeric(base.get(stat_l10, np.nan), errors="coerce")
        s20 = pd.to_numeric(base.get(stat_l20, np.nan), errors="coerce")
        pg_base = sb / gp_season
        pg10 = s10 / gp_10
        pg20 = s20 / gp_20
        rec = _blend_recency(pg_base, pg10, pg20)
        base[out_col] = rec.fillna(pg_base)

    recency_pg("xGoalsFor", "xGoalsFor_L10", "xGoalsFor_L20", "xGF_pg_recency")
    recency_pg("xGoalsAgainst", "xGoalsAgainst_L10", "xGoalsAgainst_L20", "xGA_pg_recency")

    return base


# -----------------------------
# Goalies
# -----------------------------

def load_goalies() -> pd.DataFrame:
    print(" Loading MoneyPuck goalies (season + L10 + L20)...")
    g0 = _load_mp_csv(MP_GOALIES_SEASON, "goalies seasonSummary")
    if g0.empty:
        raise RuntimeError("MoneyPuck goalies seasonSummary is empty.")

    if not {"name", "team", "situation", "icetime", "xGoals", "goals"}.issubset(g0.columns):
        raise ValueError("goalies.csv missing required columns.")

    g0["team"] = g0["team"].apply(normalize_team)
    g0["player_clean"] = g0["name"].apply(clean_name)

    base = g0[g0["situation"] == "all"].copy()
    if base.empty:
        base = g0[g0["situation"] == "5on5"].copy()

    base = base.sort_values(
        ["player_clean", "team", "icetime"], ascending=[True, True, False]
    ).drop_duplicates(subset=["player_clean", "team"], keep="first")

    base = base[[c for c in base.columns if c != "situation"]].copy()

    g10 = _load_mp_csv(MP_GOALIES_L10, "goalies L10")
    if not g10.empty:
        g10["team"] = g10["team"].apply(normalize_team)
        g10["player_clean"] = g10["name"].apply(clean_name)
        b10 = g10[g10["situation"] == "all"].copy()
        if b10.empty:
            b10 = g10[g10["situation"] == "5on5"].copy()
        b10 = b10.sort_values(
            ["player_clean", "team", "icetime"], ascending=[True, True, False]
        ).drop_duplicates(subset=["player_clean", "team"], keep="first")
        b10 = b10[[c for c in b10.columns if c != "situation"]]
        b10 = b10.rename(columns={c: f"{c}_L10" for c in b10.columns if c not in {"player_clean", "team"}})
        base = base.merge(b10, how="left", on=["player_clean", "team"])

    g20 = _load_mp_csv(MP_GOALIES_L20, "goalies L20")
    if not g20.empty:
        g20["team"] = g20["team"].apply(normalize_team)
        g20["player_clean"] = g20["name"].apply(clean_name)
        b20 = g20[g20["situation"] == "all"].copy()
        if b20.empty:
            b20 = g20[g20["situation"] == "5on5"].copy()
        b20 = b20.sort_values(
            ["player_clean", "team", "icetime"], ascending=[True, True, False]
        ).drop_duplicates(subset=["player_clean", "team"], keep="first")
        b20 = b20[[c for c in b20.columns if c != "situation"]]
        b20 = b20.rename(columns={c: f"{c}_L20" for c in b20.columns if c not in {"player_clean", "team"}})
        base = base.merge(b20, how="left", on=["player_clean", "team"])

    if "xGoals" in base.columns and "games_played" in base.columns:
        xga_pg = pd.to_numeric(base["xGoals"], errors="coerce") / pd.to_numeric(base["games_played"], errors="coerce").replace(0, np.nan)
        xga_pg_10 = pd.to_numeric(base.get("xGoals_L10", np.nan), errors="coerce") / pd.to_numeric(base.get("games_played_L10", base["games_played"]), errors="coerce").replace(0, np.nan)
        xga_pg_20 = pd.to_numeric(base.get("xGoals_L20", np.nan), errors="coerce") / pd.to_numeric(base.get("games_played_L20", base["games_played"]), errors="coerce").replace(0, np.nan)
        base["xGA_pg_recency_goalie"] = _blend_recency(xga_pg, xga_pg_10, xga_pg_20).fillna(xga_pg)

    return base


# -----------------------------
# Lines env
# -----------------------------

def load_lines_env() -> pd.DataFrame:
    print(" Loading MoneyPuck lines (5v5 line/pairing env)...")
    dfl = _load_mp_csv(MP_LINES, "lines")
    if dfl.empty:
        print(" lines.csv empty/missing - line env will be mostly NaN.")
        return pd.DataFrame()

    required = {"team", "position", "situation", "icetime"}
    if not required.issubset(dfl.columns):
        print(" lines.csv missing expected columns; skipping line env.")
        return pd.DataFrame()

    dfl["team"] = dfl["team"].apply(normalize_team)
    dfl = dfl[dfl["situation"] == "5on5"].copy()
    dfl = dfl.sort_values(["team", "position", "icetime"], ascending=[True, True, False])

    roles = {}

    def assign_roles(sub: pd.DataFrame, prefix: str, max_n: int):
        idxs = list(sub.index)
        out = {}
        for i, idx in enumerate(idxs[:max_n]):
            out[idx] = f"{prefix}{i+1}"
        return out

    for (team, pos), sub in dfl.groupby(["team", "position"], sort=False):
        if pos == "line":
            r = assign_roles(sub, "L", 4)
        elif pos == "pairing":
            r = assign_roles(sub, "P", 3)
        else:
            continue
        roles.update(r)

    dfl["role"] = dfl.index.to_series().map(roles)
    dfl = dfl[dfl["role"].notna()].copy()
    dfl["env_key"] = dfl["team"] + "_" + dfl["role"]
    return dfl


# -----------------------------
# PP usage (optional)
# -----------------------------

def load_pp_usage(path: Path):
    try:
        dfpp = pd.read_excel(path)
    except Exception:
        print(" PP usage file missing or unreadable; PP_TOI will stay NaN.")
        return None

    dfpp.columns = [c.strip() for c in dfpp.columns]
    cand_player = [c for c in dfpp.columns if c.lower() in {"player", "name"}]
    cand_team = [c for c in dfpp.columns if "team" in c.lower()]
    cand_pp = [c for c in dfpp.columns if "pp" in c.lower() and "toi" in c.lower()]

    if not (cand_player and cand_team and cand_pp):
        print(" PP usage file format not recognized; skipping PP merge.")
        return None

    pcol = cand_player[0]
    tcol = cand_team[0]
    ppcol = cand_pp[0]

    dfpp["player_clean"] = dfpp[pcol].apply(clean_name)
    dfpp["TEAM"] = dfpp[tcol].apply(normalize_team)
    dfpp = dfpp[["player_clean", "TEAM", ppcol]].rename(columns={ppcol: "PP_TOI"})
    return dfpp


# -----------------------------
# Derived rates / multipliers
# -----------------------------

def add_rates_and_roles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_forward"] = df["base_pos"].isin(["C", "W"])
    df["is_defense"] = df["base_pos"].eq("D")

    toi_sec = pd.to_numeric(df.get("icetime", 0.0), errors="coerce")
    hours = toi_sec / 3600.0
    hours = hours.replace(0, np.nan)

    def rate(col):
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        total = pd.to_numeric(df[col], errors="coerce")
        r = total / hours
        return r.replace([np.inf, -np.inf], np.nan)

    sk_mask = df["is_forward"] | df["is_defense"]

    for col, out_col in [
        ("I_F_xGoals", "xG_per60"),
        ("I_F_shotsOnGoal", "SOG_per60"),
        ("I_F_shotAttempts", "CF_per60"),
    ]:
        r = rate(col)
        df[out_col] = np.where(sk_mask, r, np.nan)

    xgf_team = pd.to_numeric(df.get("xGoalsFor_teamenv", 0.0), errors="coerce")
    xga_team = pd.to_numeric(df.get("xGoalsAgainst_teamenv", 0.0), errors="coerce")

    league_xgf = xgf_team.replace(0, np.nan).mean()
    league_xga = xga_team.replace(0, np.nan).mean()

    if league_xgf and not np.isnan(league_xgf):
        df["team_offense_mult"] = np.where(
            xgf_team > 0, xgf_team / league_xgf, 1.0
        )
    else:
        df["team_offense_mult"] = 1.0

    if league_xga and not np.isnan(league_xga):
        df["team_defense_mult"] = np.where(
            xga_team > 0, xga_team / league_xga, 1.0
        )
    else:
        df["team_defense_mult"] = 1.0

    line_xgf = pd.to_numeric(df.get("xGoalsFor_lineenv", 0.0), errors="coerce")
    df["line_offense_mult"] = np.where(
        (line_xgf > 0) & (xgf_team > 0),
        line_xgf / xgf_team.replace(0, np.nan),
        1.0,
    )

    df["line_strength_raw"] = (
        df["xG_per60"].fillna(0)
        + df.get("SOG_per60", 0).fillna(0) * 0.25
    )

    return df


# -----------------------------
# Main merge
# -----------------------------

def build_merged_player_pool() -> pd.DataFrame:
    df_rw = load_rotowire(RW_FILE)
    df_skaters = load_skaters()
    df_teams = load_teams()
    df_goalies = load_goalies()
    df_lines = load_lines_env()
    df_pp = load_pp_usage(PP_FILE)

    print(" Merging skater-level stats...")
    df = df_rw.merge(
        df_skaters,
        how="left",
        left_on=["player_clean", "TEAM"],
        right_on=["player_clean", "team"],
        suffixes=("", "_mp"),
    )
    df.drop(columns=["team"], inplace=True, errors="ignore")

    if not df_lines.empty:
        print(" Merging line / pairing environment...")
        rename_map = {}
        for col in df_lines.columns:
            if col in {"xGoalsFor", "xGoalsAgainst", "shotsOnGoalFor", "shotsOnGoalAgainst"}:
                rename_map[col] = col + "_lineenv"
        df_lines2 = df_lines.rename(columns=rename_map)
        (
            df_lines2,
            how="left",
            on="env_key",
            suffixes=("", "_lineenv_extra"),
        )
    else:
        print(" No line env; env_key will exist but line env stats are NaN.")

    if not df_teams.empty:
        print(" Merging team-level environment...")
        df = df.merge(
            df_teams,
            how="left",
            left_on="TEAM",
            right_on="team",
            suffixes=("", "_teamenv"),
        )
        df.drop(columns=["team"], inplace=True, errors="ignore")

        rename_teamenv = {}
        for col in ["xGoalsFor", "xGoalsAgainst", "shotsOnGoalFor", "shotsOnGoalAgainst"]:
            if col in df.columns:
                rename_teamenv[col] = col + "_teamenv"
        if rename_teamenv:
            df.rename(columns=rename_teamenv, inplace=True)

        print(" Merging opponent team environment...")
        df_teams_opp = df_teams.rename(
            columns={
                "team": "team_opp",
                "games_played": "games_played_teamenv_opp",
                "xGoalsFor": "xGoalsFor_teamenv_opp",
                "xGoalsAgainst": "xGoalsAgainst_teamenv_opp",
                "shotsOnGoalFor": "shotsOnGoalFor_teamenv_opp",
                "shotAttemptsFor": "shotAttemptsFor_teamenv_opp",
                "shotsOnGoalAgainst": "shotsOnGoalAgainst_teamenv_opp",
                "shotAttemptsAgainst": "shotAttemptsAgainst_teamenv_opp",
                "xGF_pg_recency": "xGF_pg_recency_opp",
                "xGA_pg_recency": "xGA_pg_recency_opp",
            }
        )
        df = df.merge(
            df_teams_opp,
            how="left",
            left_on="OPP",
            right_on="team_opp",
        )
        df.drop(columns=["team_opp"], inplace=True, errors="ignore")
    else:
        print(" No team env; using neutral multipliers later.")

    print(" Merging goalie stats...")
    df = df.merge(
        df_goalies,
        how="left",
        left_on=["player_clean", "TEAM"],
        right_on=["player_clean", "team"],
        suffixes=("", "_goalie"),
    )
    df.drop(columns=["team"], inplace=True, errors="ignore")
    
    # -----------------------------
    # PP merge
    # -----------------------------
    if df_pp is not None:
        print(" Merging PP usage (PP_TOI)...")
        df = df.merge(
            df_pp,
            how="left",
            on=["player_clean", "TEAM"],
        )
    else:
        df["PP_TOI"] = np.nan
    
    # -----------------------------------------------------
    # FIX POSITIONS BASED ON MONEYPUCK GOALIE STATS
    # -----------------------------------------------------
    goalie_cols = [
        "xGoals",                 # season xGoals
        "goals",                  # actual GA
        "games_played_goalie",    # MoneyPuck GP
        "goalie_xGoals",          # some MP versions
        "xGoals_L10", "xGoals_L20"
    ]
    
    for col in goalie_cols:
        if col in df.columns:
            df.loc[
                df[col].notna() & (pd.to_numeric(df[col], errors="coerce") > 0),
                "base_pos"
            ] = "G"
    
    # Refresh goalie flag
    df["is_goalie"] = df["base_pos"].astype(str).str.upper().eq("G")

    print(" Adding derived per-60 rates and multipliers...")
    df = add_rates_and_roles(df)

    missing_mp = df["base_pos"].isin(["C", "W", "D"]) & df.get("I_F_xGoals", pd.Series(index=df.index)).isna()
    if missing_mp.any():
        print(f" {missing_mp.sum()} skaters missing MoneyPuck advanced stats (name or team mismatch).")

    missing_goalies = df["is_goalie"] & df.get("xGoals", pd.Series(index=df.index)).isna()
    if missing_goalies.any():
        print(f" {missing_goalies.sum()} goalies missing MoneyPuck goalie stats.")

    return df


def main():
    df = build_merged_player_pool()
    print(f" Writing merged player pool -> {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    print(" Done.")


if __name__ == "__main__":
    main()
