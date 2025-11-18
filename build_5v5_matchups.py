import pandas as pd
from pathlib import Path

# -------------------------------------------------------
# CLOUD-SAFE ROOT DIR (same folder where script lives)
# -------------------------------------------------------
APP_ROOT = Path(__file__).parent.resolve()

SCRAPED = APP_ROOT / "5v5_matchups.csv"
MERGED  = APP_ROOT / "merged_nhl_player_pool.csv"
OUTPUT  = APP_ROOT / "5v5_matchups_summary.csv"

# ---------------------------------------
# MASTER FULL TEAM NAME -> ABBREV MAP
# ---------------------------------------
TEAM_NAME_MAP = {
    "DUCKS": "ANA", "COYOTES": "ARI", "BRUINS": "BOS",
    "SABRES": "BUF", "FLAMES": "CGY", "HURRICANES": "CAR",
    "BLACKHAWKS": "CHI", "AVALANCHE": "COL", "BLUE JACKETS": "CBJ",
    "STARS": "DAL", "RED WINGS": "DET", "OILERS": "EDM",
    "PANTHERS": "FLA", "KINGS": "LAK", "WILD": "MIN",
    "CANADIENS": "MTL", "PREDATORS": "NSH", "DEVILS": "NJD",
    "ISLANDERS": "NYI", "RANGERS": "NYR", "SENATORS": "OTT",
    "FLYERS": "PHI", "PENGUINS": "PIT", "SHARKS": "SJS",
    "KRAKEN": "SEA", "BLUES": "STL", "LIGHTNING": "TBL",
    "MAPLE LEAFS": "TOR", "CANUCKS": "VAN", "KNIGHTS": "VGK",
    "CAPITALS": "WSH", "JETS": "WPG",
}

def normalize_team_name(name: str):
    name = str(name).upper().strip()
    return TEAM_NAME_MAP.get(name, None)


# --------------------------------------------------------
# Build usable line-vs-line matchup rows
# --------------------------------------------------------
def build_matchups(scraped_file: Path, merged_file: Path, output_file: Path):
    print(f" Reading scraped file -> {scraped_file}")
    s = pd.read_csv(scraped_file)

    print(f" Reading merged player pool -> {merged_file}")
    m = pd.read_csv(merged_file)

    # Normalize merged pool
    m["TEAM_KEY"] = m["TEAM"].astype(str).str.upper().str.strip()
    m["OPP_KEY"]  = m["OPP"].astype(str).str.upper().str.strip()

    # Map team → opponent
    opp_map = m.groupby("TEAM_KEY")["OPP_KEY"].first().to_dict()

    # Normalize scraped names
    s["team_raw"] = s["line_short_team"].astype(str)
    s["team"] = s["line_short_team"].apply(normalize_team_name)
    s["line"] = s["line"].astype(str).str.upper().str.strip()

    # Add opponent info based on merged pool
    s["opp_team"] = s["team"].map(opp_map)

    # Warn if mapping failed
    missing = s[s["opp_team"].isna()]
    if len(missing) > 0:
        print("\n WARNING — These rows could not map to an opponent and will be skipped:")
        print(missing[["team_raw", "team", "line"]])

    # Columns containing percentage of time vs opponent lines
    pct_cols = [
        "percent_time_vs_opp_team_line_1",
        "percent_time_vs_opp_team_line_2",
        "percent_time_vs_opp_team_line_3",
        "percent_time_vs_opp_team_line_4",
    ]

    final_rows = []

    for _, row in s.iterrows():
        team = row["team"]
        opp_team = row["opp_team"]

        if pd.isna(team) or pd.isna(opp_team):
            continue

        line = row["line"]

        # Build 4 rows (one for each opponent line)
        for idx, col in enumerate(pct_cols):
            opp_line = f"FL{idx + 1}"
            toi_pct = float(row[col])

            final_rows.append({
                "team": team,
                "line": line,
                "opp_team": opp_team,
                "opp_line": opp_line,
                "toi": toi_pct,
            })

    df_out = pd.DataFrame(final_rows)

    print(f"\n Writing final output -> {output_file}")
    df_out.to_csv(output_file, index=False)
    print(" Done — 5v5 matchup summary is ready.")


if __name__ == "__main__":
    build_matchups(SCRAPED, MERGED, OUTPUT)
