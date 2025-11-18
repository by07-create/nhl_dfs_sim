# streamlit_nhl_sim.py – Streamlit NHL DFS front-end for your existing pipeline
#
# Flow:
#   1) Upload rw-nhl-player-pool.xlsx
#   2) Click "Run full pipeline"
#   3) App runs your existing scripts in order
#   4) Shows previews + download links for key CSV outputs.

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------------------------
# Core paths – repo-relative, cloud-safe
# -------------------------------------------------
# Directory where this app file lives (repo root or subfolder)
PROJECT_DIR = Path(__file__).parent.resolve()

# Files your pipeline uses/creates (all relative to PROJECT_DIR)
RW_TARGET    = PROJECT_DIR / "rw-nhl-player-pool.xlsx"
MERGED_FILE  = PROJECT_DIR / "merged_nhl_player_pool.csv"
PROJ_FILE    = PROJECT_DIR / "nhl_player_projections.csv"
FD_PROJ_FILE = PROJECT_DIR / "nhl_fd_projections.csv"
BOOM_FILE    = PROJECT_DIR / "nhl_line_boom_model.csv"
LINEUPS_FILE = PROJECT_DIR / "nhl_fd_lineups.csv"
SIM_FILE     = PROJECT_DIR / "nhl_lineup_sim_leaderboard.csv"

# Scripts in pipeline – they must live in the same folder as this app
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

    # Show logs
    if result.stdout:
        st.text_area(
            f"{label} stdout",
            result.stdout,
            height=180,
        )
    if result.stderr:
        st.text_area(
            f"{label} stderr",
            result.stderr,
            height=160,
        )

    if result.returncode != 0:
        st.error(f"[ERROR] {label} failed (exit code {result.returncode}).")
        return False

    st.success(f"[OK] {label} completed successfully.")
    return True


def show_csv_preview(path: Path, title: str, n: int = 50):
    """Show a preview + download button for CSVs."""
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
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="NHL DFS Simulator", layout="wide")
    st.title("NHL DFS Pipeline – RW -> MoneyPuck -> Stacks -> Lineups")

    # ---------------------------------------------
    # 1) Upload RW file
    # ---------------------------------------------
    st.header("1) Upload Rotowire NHL Player Pool")

    rw_file = st.file_uploader(
        "Upload rw-nhl-player-pool.xlsx",
        type=["xlsx", "xls"],
    )

    if rw_file is not None:
        # Save uploaded RW file into PROJECT_DIR
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

    # ---------------------------------------------
    # 4) Line Boom Charts (Exact Match to Dashboard)
    # ---------------------------------------------

    import plotly.express as px
    import pandas as pd

    st.header("4) Line Boom Charts – Forward, Defense, & Combined")

    tabs = st.tabs([
        "Forward Lines",
        "Defense Stacks",
        "Combined F + D Stacks",
        "Legend / Key"
    ])

    # -------------------------------------------------
    # Load Boom Model
    # -------------------------------------------------
    if not BOOM_FILE.exists():
        df_boom = None
        for t in tabs:
            with t:
                st.warning("Run the pipeline first. Boom model not found.")
    else:
        df_boom = pd.read_csv(BOOM_FILE).copy()

        # rebuild line_role like your script does
        df_boom["line_role"] = (
            df_boom["env_key"]
            .astype(str)
            .str.upper()
            .str.extract(r"_(P\d+|L\d+)", expand=False)
            .fillna("")
        )

        # fill missing safety columns
        needed = [
            "TEAM","env_key","line_role","line_score","fwd_score","fld_score",
            "best_def_for_line","best_def_score","best_stack_score",
            "line_salary_total","matchup_mult_mean","goalie_mult","line_hdcf",
            "fld_line_hdcf","fwd_line_hdcf"
        ]
        for col in needed:
            if col not in df_boom.columns:
                df_boom[col] = None

        df_boom["best_def_score"] = df_boom["best_def_score"].fillna(0)
        df_boom["best_stack_score"] = df_boom["best_stack_score"].fillna(0)
        df_boom["line_salary_total"] = df_boom["line_salary_total"].fillna(0)

    # -------------------------------------------------
    # TAB 1 — Forward Lines
    # -------------------------------------------------
    with tabs[0]:
        st.subheader("Forward Lines – (line_score vs fwd_score)")

        if df_boom is not None:
            df_fwd = df_boom[df_boom["line_role"].str.startswith("L", na=False)].copy()

            fig1 = px.scatter(
                df_fwd,
                x="line_score",
                y="fwd_score",
                size="best_stack_score",
                color="TEAM",
                hover_name="env_key",
                title="Best Forward Lines Tonight (FWD Score vs Line Score)",
            )

            st.plotly_chart(fig1, use_container_width=True)

    # -------------------------------------------------
    # TAB 2 — Defense Stacks (Guaranteed non-empty)
    # -------------------------------------------------
    with tabs[1]:
        st.subheader("Defense Stacks – (salary vs best_def_score)")

        if df_boom is not None:
            # strict filter first
            df_def = df_boom[df_boom["line_role"].str.startswith("P", na=False)].copy()

            # fallback logic from your script
            if df_def.empty:
                df_def = df_boom[df_boom["env_key"].astype(str).str.contains("_P", na=False)].copy()

            if df_def.empty:
                df_def = df_boom.copy()

            df_def["best_def_score"] = df_def["best_def_score"].fillna(0)
            df_def["line_salary_total"] = df_def["line_salary_total"].fillna(0)

            fig2 = px.scatter(
                df_def,
                x="line_salary_total",
                y="best_def_score",
                color="TEAM",
                hover_name="env_key",
                title="Best Defense Stacks (Salary vs D-Stack Score)",
            )

            st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------------------
    # TAB 3 — Combined Stacks
    # -------------------------------------------------
    with tabs[2]:
        st.subheader("Combined F + D Stack Scores")

        if df_boom is not None:
            df_fwd = df_boom[df_boom["line_role"].str.startswith("L", na=False)].copy()

            fig3 = px.scatter(
                df_fwd,
                x="fwd_score",
                y="best_def_score",
                size="best_stack_score",
                color="TEAM",
                hover_name="env_key",
                title="Top Combined F + D Stacks (FWD Score vs DEF Stack Score)",
            )

            st.plotly_chart(fig3, use_container_width=True)

    # -------------------------------------------------
    # TAB 4 — Legend / Key (exactly like your HTML)
    # -------------------------------------------------
    with tabs[3]:
        st.subheader("Legend / Key")

        st.markdown("""
        ### Metrics
        - **line_score** – Full-line DFS score (simulation + xG + danger + matchup).
        - **fwd_score** – Forward-only ceiling rating (best for 2–3 man stacks).
        - **fld_score** – Full-line score including defense.
        - **best_def_for_line** – The defense pairing that stacks best with this forward line.
        - **best_def_score** – Synergy strength of that D stack.
        - **best_stack_score** – Combined forward + defense stack rating.
        - **goalie_mult** – Opposing goalie weakness multiplier.
        - **matchup_mult_mean** – Soft/hard matchup indicator.
        - **line_hdcf** – Recency high-danger chance rate.
        - **line_salary_total** – Combined FanDuel salary.

        ### How to read the charts
        - **Big bubbles** = bigger upside stack
        - **Top-right** = best plays (high boom + high projection)
        - **Bottom-right** = safe floors but lower upside
        - **Top-left** = low ownership leverage options
        """)



if __name__ == "__main__":
    main()
