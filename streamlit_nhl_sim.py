# streamlit_nhl_sim.py – Streamlit NHL DFS front-end for your updated pipeline
#
# Flow:
#   1) Upload rw-nhl-player-pool.xlsx
#   2) Click "Run full pipeline"
#   3) App runs your existing scripts in order (cleaned list)
#   4) Shows previews + download links for projections, lineups, sims, boom stacks.

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------------------------
# Core paths – repo-relative, cloud-safe
# -------------------------------------------------
PROJECT_DIR = Path(__file__).parent.resolve()

RW_TARGET    = PROJECT_DIR / "rw-nhl-player-pool.xlsx"
MERGED_FILE  = PROJECT_DIR / "merged_nhl_player_pool.csv"
PROJ_FILE    = PROJECT_DIR / "nhl_player_projections.csv"
FD_PROJ_FILE = PROJECT_DIR / "nhl_fd_projections.csv"
LINEUPS_FILE = PROJECT_DIR / "nhl_fd_lineups.csv"
SIM_FILE     = PROJECT_DIR / "nhl_lineup_sim_leaderboard.csv"
BOOM_FILE    = PROJECT_DIR / "nhl_line_boom_model.csv"   # from slate sim

# -------------------------------------------------
# CLEAN pipeline – only scripts that still exist
# -------------------------------------------------
SCRIPT_ORDER = [
    ("Merge RW + MoneyPuck",           "merge_nhl_data.py"),
    ("Scrape 5v5 line matchups",       "dailyfaceoff_matchups_scraper.py"),
    ("Build 5v5 matchup summary",      "build_5v5_matchups.py"),
    ("Build player projections",       "nhl_projection_engine.py"),
    ("Export projections for FD",      "nhl_export_for_fd.py"),
    ("Build FD lineups",               "nhl_fd_lineup_builder.py"),
    ("Run slate simulator",            "nhl_slate_simulator.py"),
]


def run_script(label: str, script_name: str) -> bool:
    """Run one pipeline script and display logs."""
    script_path = PROJECT_DIR / script_name
    if not script_path.exists():
        st.warning(f"[WARNING] {label}: script not found -> {script_path}")
        return False

    st.write(f"### [STEP] {label}")
    with st.spinner(f"Running {script_name}..."):
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
        )

    if result.stdout:
        st.text_area(f"{label} stdout", result.stdout, height=180)
    if result.stderr:
        st.text_area(f"{label} stderr", result.stderr, height=160)

    if result.returncode != 0:
        st.error(f"[ERROR] {label} failed (exit code {result.returncode}).")
        return False

    st.success(f"[OK] {label} completed successfully.")
    return True


def show_csv_preview(path: Path, title: str, n: int = 50):
    """Show preview & download link for CSVs."""
    if not path.exists():
        st.warning(f"[WARNING] {title}: file not found -> {path}")
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed reading {path}: {e}")
        return

    st.write(f"### {title} (top {n})")
    st.dataframe(df.head(n))

    st.download_button(
        label=f"Download {title}",
        data=df.to_csv(index=False),
        file_name=path.name,
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="NHL DFS Simulator", layout="wide")
    st.title("NHL DFS Pipeline – RW → MoneyPuck → Matchups → Lineups → Sims")

    # ---------------------------------------------
    # 1) Upload RW pool
    # ---------------------------------------------
    st.header("1) Upload Rotowire NHL Player Pool")

    rw_file = st.file_uploader("Upload rw-nhl-player-pool.xlsx", type=["xlsx", "xls"])

    if rw_file is not None:
        with open(RW_TARGET, "wb") as f:
            f.write(rw_file.getbuffer())
        st.success(f"[OK] Saved Rotowire file to: {RW_TARGET}")
    else:
        st.info("Upload the Rotowire file before running.")

    # ---------------------------------------------
    # 2) Run Full Pipeline
    # ---------------------------------------------
    st.header("2) Run Full NHL DFS Pipeline")

    if st.button("Run pipeline"):
        if not RW_TARGET.exists():
            st.error(f"Rotowire file missing: {RW_TARGET}")
        else:
            all_ok = True
            for label, script in SCRIPT_ORDER:
                if not run_script(label, script):
                    all_ok = False
                    break
            if all_ok:
                st.success("✔ Full pipeline completed successfully.")

    # ---------------------------------------------
    # 3) Output Previews
    # ---------------------------------------------
    st.header("3) Output Files")

    col1, col2 = st.columns(2)

    with col1:
        show_csv_preview(PROJ_FILE,    "Player Projections")
        show_csv_preview(FD_PROJ_FILE, "FanDuel Projections")

    with col2:
        show_csv_preview(LINEUPS_FILE, "Generated FD Lineups")
        show_csv_preview(SIM_FILE,     "Simulation Leaderboard")

    # Also show merged data after sim runs
    show_csv_preview(MERGED_FILE, "Merged RW + MoneyPuck")

    # ---------------------------------------------
    # 4) Line Boom Charts (Boom model from slate sim)
    # ---------------------------------------------
    st.header("4) Line Boom Charts (Forward, Defense, Combined)")

    tabs = st.tabs([
        "Forward Lines",
        "Defense Stacks",
        "Combined F + D",
        "Legend / Key"
    ])

    # Load boom model only if available
    if not BOOM_FILE.exists():
        df_boom = None
        for t in tabs:
            with t:
                st.warning("Run pipeline first. Boom model not found.")
        return

    df_boom = pd.read_csv(BOOM_FILE).copy()

    # Extract L1/L2/P1 tags
    df_boom["line_role"] = (
        df_boom["env_key"]
        .astype(str)
        .str.upper()
        .str.extract(r"_(P\d+|L\d+)", expand=False)
        .fillna("")
    )

    # Safety fill cols
    needed_cols = [
        "TEAM", "env_key", "line_role",
        "line_score", "fwd_score", "fld_score",
        "best_def_for_line", "best_def_score", "best_stack_score",
        "line_salary_total", "matchup_mult_mean", "goalie_mult",
        "line_hdcf", "fld_line_hdcf", "fwd_line_hdcf"
    ]
    for col in needed_cols:
        if col not in df_boom.columns:
            df_boom[col] = 0

    # TAB 1 — Forward Lines
    with tabs[0]:
        st.subheader("Forward Lines – (line_score vs fwd_score)")
        df_fwd = df_boom[df_boom["line_role"].str.startswith("L", na=False)]
        fig1 = px.scatter(
            df_fwd,
            x="line_score", y="fwd_score",
            size="best_stack_score", color="TEAM",
            hover_name="env_key",
            title="Best Forward Lines Tonight",
        )
        st.plotly_chart(fig1, use_container_width=True)

    # TAB 2 — Defense Stacks
    with tabs[1]:
        st.subheader("Defense Stacks – (salary vs best_def_score)")
        df_def = df_boom[df_boom["line_role"].str.startswith("P", na=False)]
        if df_def.empty:
            df_def = df_boom[df_boom["env_key"].astype(str).str.contains("_P")]
        if df_def.empty:
            df_def = df_boom.copy()

        fig2 = px.scatter(
            df_def,
            x="line_salary_total",
            y="best_def_score",
            color="TEAM",
            hover_name="env_key",
            title="Top Defense Pairings (D-Stack Score vs Salary)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # TAB 3 — Combined
    with tabs[2]:
        st.subheader("Combined Forward + Defense Stack Scores")

        df_fwd = df_boom[df_boom["line_role"].str.startswith("L", na=False)]
        fig3 = px.scatter(
            df_fwd,
            x="fwd_score",
            y="best_def_score",
            size="best_stack_score",
            color="TEAM",
            hover_name="env_key",
            title="Top Combined Stacks (FWD Score vs DEF Stack Score)",
        )
        st.plotly_chart(fig3, use_container_width=True)

    # TAB 4 — Legend
    with tabs[3]:
        st.subheader("Legend / Key")
        st.markdown("""
        ### Metrics
        - **line_score** – Full-line DFS ceiling model.
        - **fwd_score** – Forward-only scoring upside.
        - **fld_score** – Full line + defense.
        - **best_def_score** – Defense stack that pairs best with the forwards.
        - **best_stack_score** – Combined F + D stack strength.
        - **matchup_mult_mean** – Soft/hard matchup indicator.
        - **line_hdcf** – High-danger chance rate (recency).
        - **goalie_mult** – Opposing goalie weakness.
        - **line_salary_total** – Total FD salary of line.

        ### Chart Reading
        - **Top-right** = Elite stacks (strong projection + boom upside)
        - **Big bubbles** = Higher ceiling stack
        - **Bottom-right** = Safer / lower-owned
        - **Top-left** = Leverage plays
        """)


if __name__ == "__main__":
    main()
