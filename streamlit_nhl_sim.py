# streamlit_nhl_sim.py — Cloud NHL DFS Dashboard (Pipeline + Tables + HTML Charts)

import streamlit as st
import subprocess
import sys
from pathlib import Path
import pandas as pd
import streamlit.components.v1 as components

# ----------------------------------------------------------
# CLOUD-SAFE ROOT
# ----------------------------------------------------------
APP_ROOT = Path(__file__).parent.resolve()


# ----------------------------------------------------------
# Helpers to run scripts and show outputs
# ----------------------------------------------------------
def run_script(script_name: str) -> bool:
    """
    Execute a Python file inside the same container (Streamlit Cloud safe).
    Returns True if exit code == 0, else False.
    """
    script_path = APP_ROOT / script_name
    if not script_path.exists():
        st.error(f"❌ Script not found: {script_path}")
        return False

    st.write(f"▶ Running: `{script_name}`")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(APP_ROOT),
            check=False,           # don't raise, we handle returncode
            capture_output=True,
            text=True,
        )
    except Exception as e:
        st.error(f"❌ Failed to execute {script_name}: {e}")
        return False

    if result.stdout.strip():
        with st.expander(f"stdout: {script_name}", expanded=False):
            st.text(result.stdout)

    if result.stderr.strip():
        with st.expander(f"stderr: {script_name}", expanded=False):
            st.text(result.stderr)

    if result.returncode != 0:
        st.error(f"❌ {script_name} failed with code {result.returncode}")
        return False

    st.success(f"✅ {script_name} completed")
    return True


def show_csv_preview(path: Path, title: str, rows: int = 100):
    """Render a CSV as a dataframe with a download button, if it exists."""
    st.subheader(title)
    if not path.exists():
        st.warning(f"File not found: {path.name}")
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
        return

    st.caption(f"{path.name} — showing top {rows} rows")
    st.dataframe(df.head(rows), use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇ Download {path.name}",
        data=csv_bytes,
        file_name=path.name,
        mime="text/csv",
    )


def show_html_report(path: Path, title: str, height: int = 900):
    """Embed an HTML report (from line_boom_chart.py) inside Streamlit."""
    st.subheader(title)
    if not path.exists():
        st.warning(f"HTML report not found: {path.name}. Run the boom chart script.")
        return

    try:
        html_str = path.read_text(encoding="utf-8")
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
        return

    components.html(html_str, height=height, scrolling=True)


# ----------------------------------------------------------
# Files we care about
# ----------------------------------------------------------
RW_FILE              = APP_ROOT / "rw-nhl-player-pool.xlsx"
MERGED_FILE          = APP_ROOT / "merged_nhl_player_pool.csv"
MATCHUPS_RAW_FILE    = APP_ROOT / "5v5_matchups.csv"
MATCHUPS_SUMMARY     = APP_ROOT / "5v5_matchups_summary.csv"
PROJ_FILE            = APP_ROOT / "nhl_player_projections.csv"
LINE_MODEL_FILE      = APP_ROOT / "line_goal_model.csv"
BOOM_MODEL_FILE      = APP_ROOT / "nhl_line_boom_model.csv"
FD_PROJ_FILE         = APP_ROOT / "nhl_fd_projections.csv"
BOOM_HTML_REPORT     = APP_ROOT / "nhl_line_boom_charts.html"  # from line_boom_chart.py


# ----------------------------------------------------------
# PIPELINE ORDER (with line model, no lineup builder/sim)
# ----------------------------------------------------------
SCRIPT_ORDER = [
    ("Merge RW + MoneyPuck",          "merge_nhl_data.py"),
    ("Scrape 5v5 Matchups",           "dailyfaceoff_matchups_scraper.py"),
    ("Build 5v5 Matchup Summary",     "build_5v5_matchups.py"),
    ("Build Player Projections",      "nhl_projection_engine.py"),
    ("Build Line Goal Model",         "nhl_line_model.py"),
    ("Export Projections for FD",     "nhl_export_for_fd.py"),
    ("Build Line Boom Model",         "line_boom_model.py"),
    ("Merge Line Goal Model",         "merge_line_goal_into_projections.py"),
    # Charts are separate: line_boom_chart.py reads nhl_line_boom_model.csv
]


# ----------------------------------------------------------
# STREAMLIT PAGE LAYOUT
# ----------------------------------------------------------
st.set_page_config(
    page_title="NHL DFS Cloud Dashboard",
    layout="wide",
)

st.title("🏒 NHL DFS Cloud Dashboard")

st.markdown(
    """
This app runs your **full NHL DFS pipeline** in the cloud and shows:

- Merged Rotowire + MoneyPuck player pool  
- 5v5 line matchups & summary  
- Player projection engine output  
- **Line goal model** (lambda_total, p_goal_line, chemistry)  
- **Line boom model** (stack scores, matchup metrics)  
- FanDuel-ready projection export  
- HTML dashboard from `line_boom_chart.py` (Option B)
"""
)

tabs = st.tabs(
    [
        "1️⃣ Pipeline & Inputs",
        "2️⃣ Core Tables",
        "3️⃣ Line Models (Goal & Boom)",
        "4️⃣ Boom Charts (HTML Report)",
    ]
)

# ----------------------------------------------------------
# TAB 1 — Pipeline & Inputs
# ----------------------------------------------------------
with tabs[0]:
    st.header("📤 Upload Inputs")

    col_u1, col_u2 = st.columns(2)

    with col_u1:
        uploaded_rw = st.file_uploader(
            "Rotowire Player Pool (rw-nhl-player-pool.xlsx)",
            type=["xlsx"],
            key="rw_upload",
        )
        if uploaded_rw:
            with open(RW_FILE, "wb") as f:
                f.write(uploaded_rw.getbuffer())
            st.success(f"Saved: {RW_FILE.name}")

    with col_u2:
        uploaded_5v5 = st.file_uploader(
            "Optional 5v5 Matchups CSV (fallback for scraper)",
            type=["csv"],
            key="matchups_upload",
        )
        if uploaded_5v5:
            with open(MATCHUPS_RAW_FILE, "wb") as f:
                f.write(uploaded_5v5.getbuffer())
            st.success(f"Saved fallback: {MATCHUPS_RAW_FILE.name}")

    st.markdown("---")

    st.header("🚀 Run Full Pipeline")

    if st.button("Run Full Pipeline Now"):
        if not RW_FILE.exists():
            st.error("Rotowire file missing. Upload rw-nhl-player-pool.xlsx first.")
        else:
            for label, script in SCRIPT_ORDER:
                st.write(f"### ▶ {label}")
                ok = run_script(script)
                if not ok:
                    st.error(f"🛑 Pipeline stopped at: {label}")
                    break
            else:
                st.success("🎉 Pipeline completed successfully.")

    st.markdown("---")

    st.header("Quick Input File Check")
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.write("RW File:")
        st.code(str(RW_FILE))
        st.write("✅" if RW_FILE.exists() else "❌ Missing")

    with col_i2:
        st.write("5v5 Raw:")
        st.code(str(MATCHUPS_RAW_FILE))
        st.write("✅" if MATCHUPS_RAW_FILE.exists() else "⚠ Optional / Missing")

    with col_i3:
        st.write("Merged Pool:")
        st.code(str(MERGED_FILE))
        st.write("✅" if MERGED_FILE.exists() else "❌ Missing (run pipeline)")


# ----------------------------------------------------------
# TAB 2 — Core Tables (Merged, 5v5, Projections)
# ----------------------------------------------------------
with tabs[1]:
    st.header("📊 Core Data Tables")

    st.markdown("### A. Merged Rotowire + MoneyPuck Player Pool")
    show_csv_preview(MERGED_FILE, "Merged Player Pool")

    st.markdown("---")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("### B. Raw 5v5 Matchups")
        show_csv_preview(MATCHUPS_RAW_FILE, "5v5 Raw Matchups")
    with col_m2:
        st.markdown("### C. 5v5 Matchup Summary")
        show_csv_preview(MATCHUPS_SUMMARY, "5v5 Matchup Summary")

    st.markdown("---")

    st.markdown("### D. Player Projections")
    show_csv_preview(PROJ_FILE, "Player Projections (nhl_player_projections.csv)")


# ----------------------------------------------------------
# TAB 3 — Line Models (Goal & Boom)
# ----------------------------------------------------------
with tabs[2]:
    st.header("🎯 Line Models: Goal & Boom")

    col_l1, col_l2 = st.columns(2)

    with col_l1:
        st.markdown("### Line Goal Model")
        st.caption(
            "From nhl_line_model.py → line_goal_model.csv\n"
            "- lambda_total\n"
            "- p_goal_line\n"
            "- chemistry_score\n"
        )
        show_csv_preview(LINE_MODEL_FILE, "Line Goal Model")

    with col_l2:
        st.markdown("### Line Boom Model")
        st.caption(
            "From line_boom_model.py → nhl_line_boom_model.csv\n"
            "- fwd_score / fld_score\n"
            "- matchup_mult / line_strength_norm\n"
            "- line_matchup_strength\n"
            "- boom_score / stack metrics\n"
        )
        show_csv_preview(BOOM_MODEL_FILE, "Line Boom Model")

    st.markdown("---")

    st.markdown("### FanDuel Projection Export")
    show_csv_preview(FD_PROJ_FILE, "FD Projections (nhl_fd_projections.csv)")


# ----------------------------------------------------------
# TAB 4 — Boom Charts (HTML from line_boom_chart.py)
# ----------------------------------------------------------
with tabs[3]:
    st.header("📈 Boom Charts (HTML Report)")
    st.markdown(
        """
This tab loads the **pre-rendered Plotly dashboard** generated by:
- `line_boom_chart.py` → `nhl_line_boom_charts.html`

If you haven't run that script yet, do it once (locally or via a new button),
commit the HTML file, or add a simple runner button in this app.
"""
    )

    show_html_report(BOOM_HTML_REPORT, "NHL Line Boom Charts Report")
