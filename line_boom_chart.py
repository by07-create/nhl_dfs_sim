"""
Enhanced DFS Line + Stack Dashboard (Cloud Version)
---------------------------------------------------
Generates a single-page HTML visualization containing:
    • Best forward lines
    • Best defense stacks
    • Best combined F+D stacks
    • Matchup Softness Chart
Includes a full legend/key at the top of the page.
"""

import pandas as pd
import plotly.express as px
from pathlib import Path

# ------------------------------------------
# CLOUD-SAFE PATHS
# ------------------------------------------
PROJECT_DIR = Path(__file__).parent.resolve()
INPUT_FILE = PROJECT_DIR / "nhl_line_boom_model.csv"
OUTPUT_FILE = PROJECT_DIR / "nhl_line_boom_charts.html"

print("Chart script started")
print(f"Loading CSV: {INPUT_FILE}")

# -----------------------
# Load & Normalize Data
# -----------------------
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} rows")

# Rebuild line_role safely from env_key
df["line_role"] = (
    df["env_key"]
    .astype(str)
    .str.upper()
    .str.extract(r"_(P\d+|L\d+)", expand=False)
    .fillna("")
)

# Expected fields
needed = [
    "TEAM","env_key","line_role","line_score","fwd_score","fld_score",
    "best_def_for_line","best_def_score","best_stack_score",
    "line_salary_total","matchup_mult_mean","goalie_mult","line_hdcf",
    "fld_line_hdcf","fwd_line_hdcf","matchup_softness"
]
for col in needed:
    if col not in df.columns:
        df[col] = None

# Default values
df["best_def_score"] = df["best_def_score"].fillna(0)
df["best_stack_score"] = df["best_stack_score"].fillna(0)
df["line_salary_total"] = df["line_salary_total"].fillna(0)
df["matchup_softness"] = df["matchup_softness"].fillna(0)

print("Forward rows (L*):", len(df[df["line_role"].str.startswith("L", na=False)]))
print("Defense rows (P*):", len(df[df["line_role"].str.startswith("P", na=False)]))

# -----------------------
# Legend / Key
# -----------------------
legend_text = """
<h2>Legend / Key</h2>
<ul>
<li><b>line_score</b> – Full-line DFS score (simulation + xG + danger + matchup).</li>
<li><b>fwd_score</b> – Forward-only ceiling score (best for 2–3 man stacks).</li>
<li><b>fld_score</b> – Full line score (forwards + defense).</li>
<li><b>best_def_for_line</b> – The defense unit that stacks best with this forward line.</li>
<li><b>best_def_score</b> – Strength rating of that D stack.</li>
<li><b>best_stack_score</b> – fwd_score + best_def_score.</li>
<li><b>goalie_mult</b> – Opposing goalie weakness multiplier (>1 means favorable).</li>
<li><b>matchup_mult_mean</b> – How soft/hard matchup is (>1 means easier).</li>
<li><b>line_hdcf</b> – Recency high-danger chance rate.</li>
<li><b>line_salary_total</b> – Combined FanDuel salary for this line.</li>
<li><b>matchup_softness</b> – Negative = soft matchup, Positive = tough matchup.</li>
</ul>
<hr>
"""

# -----------------------
# Chart 1 – Best Forward Lines
# -----------------------
df_fwd = df[df["line_role"].str.startswith("L", na=False)].copy()

fig1 = px.scatter(
    df_fwd,
    x="line_score",
    y="fwd_score",
    size="best_stack_score",
    color="TEAM",
    hover_name="env_key",
    title="Best Forward Lines (FWD Score vs Line Score)",
)

# -----------------------
# Chart 2 – Best Defense Stacks
# -----------------------
df_def = df[df["line_role"].str.startswith("P", na=False)].copy()

if df_def.empty:
    print("No P-lines detected via line_role. Scanning env_key instead.")
    df_def = df[df["env_key"].astype(str).str.contains("_P", na=False)].copy()

if df_def.empty:
    print("WARNING: No defense rows found. Using entire DF as fallback.")
    df_def = df.copy()

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

# -----------------------
# Chart 3 – Combined F + D Stack Scores
# -----------------------
fig3 = px.scatter(
    df_fwd,
    x="fwd_score",
    y="best_def_score",
    size="best_stack_score",
    color="TEAM",
    hover_name="env_key",
    title="Top Combined F+D Stacks (FWD Score vs DEF Score)",
)

# ===========================================================
# Chart 4 – Matchup Softness vs Line Score
# ===========================================================
fig4 = px.scatter(
    df_fwd,
    x="matchup_softness",
    y="line_score",
    size="best_stack_score",
    color="TEAM",
    hover_name="env_key",
    title="Matchup Softness vs Line Score (Soft < 0 | Hard > 0)",
)
fig4.update_layout(
    xaxis_title="Matchup Softness (negative = soft matchup)",
    yaxis_title="Line Score"
)

# ===========================================================
# Summary Tables (Top FWD, Top DEF, Top STACKS)
# ===========================================================
def build_summary_table(df, cols, title):
    out = df[cols].copy()
    out = out.to_html(index=False, justify="center", float_format="%.2f")
    return f"<h3>{title}</h3>" + out + "<br>"

summary_html = "<h2>Top Lines & Stacks Summary</h2>"

# --- Top 10 Forward Lines ---
df_top_fwd = (
    df[df["line_role"].str.startswith("L", na=False)]
    .sort_values("fwd_score", ascending=False)
    .head(10)
)
summary_html += build_summary_table(
    df_top_fwd,
    ["TEAM", "env_key", "fwd_score", "line_score", "matchup_mult_mean", "goalie_mult"],
    "Top 10 Forward Lines"
)

# --- Top 10 Defense Pairings ---
df_top_def = (
    df[df["line_role"].str.startswith("P", na=False)]
    .sort_values("fld_score", ascending=False)
    .head(10)
)
summary_html += build_summary_table(
    df_top_def,
    ["TEAM", "env_key", "fld_score", "line_salary_total", "matchup_mult_mean"],
    "Top 10 Defense Pairings"
)

# --- Top 10 Full Stacks (F + D) ---
df_top_stack = (
    df[df["line_role"].str.startswith("L", na=False)]
    .sort_values("best_stack_score", ascending=False)
    .head(10)
)
summary_html += build_summary_table(
    df_top_stack,
    ["TEAM", "env_key", "best_stack_score", "fwd_score", "best_def_for_line", "best_def_score"],
    "Top 10 Forward + Defense Stacks"
)


# -----------------------
# Build One-Page HTML Output
# -----------------------
print(f"Saving dashboard to: {OUTPUT_FILE}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("<html><body>")
    f.write(legend_text)
    f.write("<hr>")
    f.write(summary_html)
    f.write("<hr>")
    f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write("<hr>")
    f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
    f.write("<hr>")
    f.write(fig3.to_html(full_html=False, include_plotlyjs=False))
    f.write("<hr>")
    f.write(fig4.to_html(full_html=False, include_plotlyjs=False))
    f.write("</body></html>")

print("Dashboard written successfully.")
print(f"Output: {OUTPUT_FILE}")
