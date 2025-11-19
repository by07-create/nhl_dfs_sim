# streamlit_nhl_sim.py – Streamlit NHL DFS front-end (CLOUD-SAFE + UPLOAD MODE)

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------------------------
# Cloud-safe project root (NO WINDOWS PATHS)
# -------------------------------------------------
PROJECT_DIR = Path(__file__).parent.resolve()

# Output / input files generated during pipeline
RW_TARGET    = PROJECT_DIR / "rw-nhl-player-pool.xlsx"
MERGED_FILE  = PROJECT_DIR / "merged_nhl_player_pool.csv"
PROJ_FILE    = PROJECT_DIR / "nhl_player_projections.csv"
FD_PROJ_FILE = PROJECT_DIR / "nhl_fd_projections.csv"
BOOM_FILE    = PROJECT_DIR / "nhl_line_boom_model.csv"
LINEUPS_FILE = PROJECT_DIR / "nhl_fd_lineups.csv"
SIM_FILE     = PROJECT_DIR / "nhl_lineup_sim_leaderboard.csv"

# -------------------------------------------------
# Order the pipeline correctly and safely
# -------------------------------------------------
# NOTE: DailyFaceoff may fail in the cloud; fallback needed in the script itself.
SCRIPT_ORDER = [
    ("Merge RW + MoneyPuck",           "merge_nhl_data.py"),
    ("Scrape 5v5 line matchups",       "dailyfaceoff_matchups_scraper.py"),
    ("Build 5v5 matchup summary",      "build_5v5_matchups.py"),
    ("Build player projections",       "nhl_projection_engine.py"),
    ("Export projections for FD",      "nhl_export_for_fd.py"),
    ("Build line boom model",          "line_boom_model.py"),   # ← correct order
    ("Merge line goal model",          "merge_line_goal_into_projections.py"),
]

# -------------------------------------------------
# Safe script runner (no Windows dependencies)
# -------------------------------------------------
def run_script(label: str, script_name: str) -> bool:
    script_path = PROJECT_DIR / script_name
    if not script_path.exists():
        st.warning(f"[WARNING] {label}: script not found -> {script_path}")
        return False

    st.write(f"### [STEP] {label}")
    with st.spinner(f"Running {script_name}..."):
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True
        )

    # Show logs
    if result.stdout.strip():
        st.text_area(f"{label} stdout", result.stdout, height=200)

    if result.stderr.strip():
        st.text_area(f"{label} stderr", result.stderr, height=200)

    if result.returncode != 0:
        st.error(f"[ERROR] {label} failed (exit code {result.returncode}).")
        return False

    st.success(f"[OK] {label} completed successfully.")
    return True


# -------------------------------------------------
# CSV preview widget
# -------------------------------------------------
def show_csv_preview(path: Path, title: str, n: int = 50):
    if not path.exists():
        st.warning(f"[WARNING] {title}: file not found -> {path}")
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
        mime="text/csv"
    )


# -------------------------------------------------
# MAIN STREAMLIT APP
# -------------------------------------------------
def main():
    st.set_page_config(page_title="NHL DFS Simulator", layout="wide")
    st.title("NHL DFS Pipeline – RW → MoneyPuck → Lines → Lineups")

    # -------------------------------------------------
    # 1) Upload Rotowire File
    # -------------------------------------------------
    st.header("1) Upload Rotowire Player Pool (XLSX)")
    rw_file = st.file_uploader(
        "Upload rw-nhl-player-pool.xlsx",
        type=["xlsx", "xls"]
    )

    if rw_file:
        with open(RW_TARGET, "wb") as f:
            f.write(rw_file.getbuffer())
        st.success(f"[OK] Saved Rotowire file to: {RW_TARGET}")
    else:
        st.info("Upload the Rotowire file before running the pipeline.")

    # -------------------------------------------------
    # 2) Run Full Pipeline
    # -------------------------------------------------
    st.header("2) Run Full Pipeline")

    if st.button("Run Pipeline"):
        if not RW_TARGET.exists():
            st.error("Missing Rotowire file. Upload before running.")
        else:
            all_ok = True
            for label, script in SCRIPT_ORDER:
                if not run_script(label, script):
                    all_ok = False
                    break

            if all_ok:
                st.success("🎉 Pipeline completed successfully!")

    # -------------------------------------------------
    # 3) Output Previews
    # -------------------------------------------------
    st.header("3) Output Files")

    col1, col2 = st.columns(2)
    with col1:
        show_csv_preview(PROJ_FILE,    "Player Projections")
        show_csv_preview(FD_PROJ_FILE, "FanDuel Projections")
        show_csv_preview(BOOM_FILE,    "Line Boom Model")

    with col2:
        show_csv_preview(MERGED_FILE,  "Merged RW + MoneyPuck Data")

    # -------------------------------------------------
    # 4) Boom Model Charts
    # -------------------------------------------------
    st.header("4) Line Boom Charts – Forward / Defense / Combined")

    tabs = st.tabs([
        "Forward Lines",
        "Defense Stacks",
        "Combined F+D",
        "Legend"
    ])

    if not BOOM_FILE.exists():
        for t in tabs:
            with t:
                st.warning("Run pipeline first. Boom model not found.")
        return

    df = pd.read_csv(BOOM_FILE).copy()

    # rebuild line_role reliably
    df["line_role"] = (
        df["env_key"]
        .astype(str)
        .str.upper()
        .str.extract(r"_(P\d+|L\d+)", expand=False)
        .fillna("")
    )

    # TAB 1 — Forward Lines
    with tabs[0]:
        df_fwd = df[df["line_role"].str.startswith("L", na=False)].copy()
        fig = px.scatter(
            df_fwd,
            x="line_score",
            y="fwd_score",
            size="best_stack_score",
            color="TEAM",
            hover_name="env_key",
            title="Forward Lines – Line Score vs FWD Score"
        )
        st.plotly_chart(fig, use_container_width=True)

    # TAB 2 — Defense Stacks
    with tabs[1]:
        df_def = df[df["line_role"].str.startswith("P", na=False)].copy()
        if df_def.empty:
            df_def = df[df["env_key"].str.contains("_P", na=False)].copy()
        fig = px.scatter(
            df_def,
            x="line_salary_total",
            y="best_def_score",
            color="TEAM",
            hover_name="env_key",
            title="Defense Stacks – Salary vs D-Stack Score"
        )
        st.plotly_chart(fig, use_container_width=True)

    # TAB 3 — Combined F + D
    with tabs[2]:
        df_fwd = df[df["line_role"].str.startswith("L", na=False)]
        fig = px.scatter(
            df_fwd,
            x="fwd_score",
            y="best_def_score",
            size="best_stack_score",
            color="TEAM",
            hover_name="env_key",
            title="Best Combined F + D Stacks"
        )
        st.plotly_chart(fig, use_container_width=True)

    # TAB 4 — Legend
    with tabs[3]:
        st.markdown("""
        ### Legend / Key
        - **line_score** – Full-line DFS composite score  
        - **fwd_score** – Forward-only ceiling  
        - **fld_score** – Full-line (F + D) scoring  
        - **best_def_score** – Strength of best pairing with this line  
        - **best_stack_score** – FWD + D-stack synergy  
        - **matchup_mult_mean** – Soft/hard matchup indicator  
        - **goalie_mult** – Opposing goalie weakness multiplier  
        """)



if __name__ == "__main__":
    main()
