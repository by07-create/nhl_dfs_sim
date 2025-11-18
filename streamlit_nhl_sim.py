# app.py – Streamlit NHL DFS front-end for your existing pipeline
#
# Flow:
#   1) Upload rw-nhl-player-pool.xlsx
#   2) Click "Run full pipeline"
#   3) App runs your existing scripts in order
#   4) Shows previews + download links for key CSV outputs.

import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

# -------------------------------------------------
# Core paths – match your existing scripts
# -------------------------------------------------
PROJECT_DIR = Path(r"C:\Users\b-yan\OneDrive\Documents\Bo - Python Apps\NHL Simulator\streamlit")
DOWNLOADS_DIR = Path.home() / "Downloads"

# Files your pipeline already uses/creates
RW_TARGET    = DOWNLOADS_DIR / "rw-nhl-player-pool.xlsx"
MERGED_FILE  = PROJECT_DIR / "merged_nhl_player_pool.csv"
PROJ_FILE    = PROJECT_DIR / "nhl_player_projections.csv"
FD_PROJ_FILE = PROJECT_DIR / "nhl_fd_projections.csv"
BOOM_FILE    = PROJECT_DIR / "nhl_line_boom_model.csv"
LINEUPS_FILE = PROJECT_DIR / "nhl_fd_lineups.csv"
SIM_FILE     = PROJECT_DIR / "nhl_lineup_sim_leaderboard.csv"

# Scripts in pipeline
SCRIPT_ORDER = [
    ("Merge RW + MoneyPuck",           "merge_nhl_data.py"),
    ("Scrape 5v5 line matchups",       "dailyfaceoff_matchups_scraper.py"),
    ("Build 5v5 matchup summary",      "build_5v5_matchups.py"),
    ("Build player projections",       "nhl_projection_engine.py"),
    ("Export projections for FD",      "nhl_export_for_fd.py"),
    ("Merge line goal model",          "merge_line_goal_into_projections.py"),
    ("Build line boom model",          "nhl_line_model.py"),
    ("Build FD lineups",               "nhl_fd_lineup_builder.py"),
    ("Run slate simulator",            "nhl_slate_simulator.py"),
    ("Build boom stacks",              "line_boom_model.py"),
]


def run_script(label: str, script_name: str) -> bool:
    """Run one pipeline script and display logs."""
    script_path = PROJECT_DIR / script_name
    if not script_path.exists():
        st.warning(f"[WARNING] {label}: script not found → {script_path}")
        return False

    st.write(f"### [STEP] {label}")
    with st.spinner(f"Running {script_name}..."):
        result = subprocess.run(
            ["python", str(script_path)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
        )

    # Show logs
    if result.stdout:
        st.text_area(
            f"{label} – stdout",
            result.stdout,
            height=180,
        )
    if result.stderr:
        st.text_area(
            f"{label} – stderr",
            result.stderr,
            height=160,
        )

    if result.returncode != 0:
        st.error(f"[ERROR] {label} failed (exit code {result.returncode}).")
        return False

    st.success(f"[OK] {label} completed successfully.")
    return True


def show_csv_preview(path: Path, title: str, n: int = 50):
    """Shows a preview + download button for CSVs."""
    if not path.exists():
        st.warning(f"[WARNING] {title}: file not found → {path}")
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
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
    st.title("NHL DFS Pipeline – RW → MoneyPuck → Stacks → Lineups")

    # ---------------------------------------------
    # 1) Upload RW file
    # ---------------------------------------------
    st.header("1) Upload Rotowire NHL Player Pool")

    rw_file = st.file_uploader(
        "Upload rw-nhl-player-pool.xlsx",
        type=["xlsx", "xls"],
    )

    if rw_file is not None:
        RW_TARGET.parent.mkdir(parents=True, exist_ok=True)
        with open(RW_TARGET, "wb") as f:
            f.write(rw_file.getbuffer())
        st.success(f"[OK] Saved Rotowire file to: {RW_TARGET}")
    else:
        st.info("Upload the Rotowire file before running the pipeline.")

    # ---------------------------------------------
    # 2) Run Full Pipeline
    # ---------------------------------------------
    st.header("2) Run Full NHL DFS Pipeline")

    if st.button("Run pipeline"):
        if not RW_TARGET.exists():
            st.error(f"Rotowire file not found at {RW_TARGET}. Upload first.")
        else:
            all_ok = True
            for label, script in SCRIPT_ORDER:
                if not run_script(label, script):
                    all_ok = False
                    break
            if all_ok:
                st.success("[DONE] Full pipeline completed successfully.")

    # ---------------------------------------------
    # 3) Output Previews
    # ---------------------------------------------
    st.header("3) Output Files")

    col1, col2 = st.columns(2)

    with col1:
        show_csv_preview(PROJ_FILE,    "Player Projections")
        show_csv_preview(FD_PROJ_FILE, "FanDuel Projections")
        show_csv_preview(BOOM_FILE,    "Line Boom Model")

    with col2:
        show_csv_preview(LINEUPS_FILE, "Generated FD Lineups")
        show_csv_preview(SIM_FILE,     "Simulation Leaderboard")
        show_csv_preview(MERGED_FILE,  "Merged RW + MoneyPuck")


if __name__ == "__main__":
    main()
