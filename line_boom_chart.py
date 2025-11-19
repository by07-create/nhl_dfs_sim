# line_boom_chart.py — 3-Panel Stack Dashboard (Cloud-Safe)

import pandas as pd
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ----------------------------------------------------------
# Cloud-safe paths
# ----------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.resolve()
INPUT_FILE = PROJECT_DIR / "nhl_line_boom_model.csv"
OUTPUT_HTML = PROJECT_DIR / "nhl_line_boom_charts.html"

print(f" Loading line model -> {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"   Loaded {len(df)} line rows.")

# ----------------------------------------------------------
# Ensure numeric types
# ----------------------------------------------------------
numeric_cols = [
    "fwd_score",
    "line_score",
    "line_salary_total",
    "best_def_score",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        print(f" ⚠ Warning: column '{col}' not found in line model.")

df = df.dropna(
    subset=["fwd_score", "line_score", "line_salary_total", "best_def_score"],
    how="all"
)
print(f"   After cleaning, {len(df)} rows remain.")

# ----------------------------------------------------------
# Simple team color mapping
# ----------------------------------------------------------
if "TEAM" not in df.columns:
    raise ValueError("Expected 'TEAM' column in nhl_line_boom_model.csv for coloring.")

teams = sorted(df["TEAM"].astype(str).unique())
palette = px.colors.qualitative.Plotly
color_map = {team: palette[i % len(palette)] for i, team in enumerate(teams)}

def team_color(series):
    return series.astype(str).map(color_map)

# Marker size (scaled by salary)
if "line_salary_total" in df.columns:
    sal = df["line_salary_total"].fillna(0)
    if sal.max() > sal.min():
        size_scaled = 8 + (sal - sal.min()) / (sal.max() - sal.min()) * 20
    else:
        size_scaled = pd.Series(12, index=df.index)
else:
    size_scaled = pd.Series(12, index=df.index)

env_text = (
    df["TEAM"].astype(str) + " " + df["env_key"].astype(str)
    if "env_key" in df.columns
    else df["TEAM"].astype(str)
)

# ----------------------------------------------------------
# Create subplots: 3 stacked rows
# ----------------------------------------------------------
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=False,
    vertical_spacing=0.1,
    subplot_titles=(
        "1. Forward Stack Score vs Overall Line Score",
        "2. Salary vs Best Defense Stack Score (Defense-Only)",
        "3. Best Defense Score vs Forward Stack Score",
    ),
)

# ----------------------------------------------------------
# Row 1 — fwd_score vs line_score
# ----------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=df["line_score"],
        y=df["fwd_score"],
        mode="markers",
        marker=dict(
            size=size_scaled,
            color=team_color(df["TEAM"]),
            sizemode="diameter",
            opacity=0.8,
            line=dict(width=0.5, color="black"),
        ),
        text=env_text,
        hovertemplate=(
            "TEAM/Line: %{text}<br>"
            "Fwd Score: %{y:.2f}<br>"
            "Line Score: %{x:.2f}<br>"
            "Salary Total: %{customdata[0]:.0f}<extra></extra>"
        ),
        customdata=df[["line_salary_total"]].values,
        name="Fwd vs Line Score",
    ),
    row=1,
    col=1,
)

fig.update_xaxes(title_text="Overall Line Score (line_score)", row=1, col=1)
fig.update_yaxes(title_text="Forward Stack Score (fwd_score)", row=1, col=1)

fig.add_annotation(
    row=1,
    col=1,
    x=0,
    y=1.15,
    xref="x domain",
    yref="y domain",
    text=(
        "Best stacks → upper-right (high fwd_score & high line_score). "
        "Upper-left = strong line but weaker forwards; lower-right = strong forwards but weaker overall line."
    ),
    showarrow=False,
    font=dict(size=11),
)

# ----------------------------------------------------------
# Row 2 — DEFENSIVE PAIRINGS ONLY (P1, P2, P3...) + None fix
# ----------------------------------------------------------

# 1) Primary: line_role starts with P
df_def = df[df["line_role"].astype(str).str.startswith("P", na=False)].copy()

# 2) Try env_key fallback
if df_def.empty:
    print("⚠ No P-lines detected via line_role. Scanning env_key instead.")
    df_def = df[df["env_key"].astype(str).str.contains("_P", na=False)].copy()

# 3) Final fallback: use full DF (prevents blank chart)
if df_def.empty:
    print("⚠ WARNING: No defense rows found at all. Using entire DF as fallback.")
    df_def = df.copy()

# --- 🔥 Critical patch: convert "None" → NaN → 0 ---
df_def["best_def_score"] = (
    df_def["best_def_score"]
    .replace("None", None)
    .astype(float)
    .fillna(0)
)

df_def["line_salary_total"] = (
    df_def["line_salary_total"]
    .replace("None", None)
    .astype(float)
    .fillna(0)
)

fig.add_trace(
    go.Scatter(
        x=df_def["line_salary_total"],
        y=df_def["best_def_score"],
        mode="markers",
        marker=dict(
            size=10,
            color=team_color(df_def["TEAM"]),
            opacity=0.85,
            line=dict(width=0.5, color="black"),
        ),
        text=df_def["TEAM"].astype(str) + " " + df_def["env_key"].astype(str),
        hovertemplate=(
            "TEAM/Pair: %{text}<br>"
            "Total Salary: %{x:.0f}<br>"
            "Best Def Score: %{y:.2f}<extra></extra>"
        ),
        name="Salary vs Best D Score",
    ),
    row=2,
    col=1,
)

fig.update_xaxes(title_text="Total Line Salary (line_salary_total)", row=2, col=1)
fig.update_yaxes(title_text="Best Defense Stack Score (best_def_score)", row=2, col=1)

# ----------------------------------------------------------
# Row 3 — best_def_score vs fwd_score
# ----------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=df["fwd_score"],
        y=df["best_def_score"],
        mode="markers",
        marker=dict(
            size=10,
            color=team_color(df["TEAM"]),
            opacity=0.85,
            line=dict(width=0.5, color="black"),
        ),
        text=env_text,
        hovertemplate=(
            "TEAM/Line: %{text}<br>"
            "Best Def Score: %{y:.2f}<br>"
            "Fwd Score: %{x:.2f}<extra></extra>"
        ),
        name="Best D vs Fwd Score",
    ),
    row=3,
    col=1,
)

fig.update_xaxes(title_text="Forward Stack Score (fwd_score)", row=3, col=1)
fig.update_yaxes(title_text="Best Defense Stack Score (best_def_score)", row=3, col=1)

fig.add_annotation(
    row=3,
    col=1,
    x=0,
    y=1.15,
    xref="x domain",
    yref="y domain",
    text=(
        "Best combined stacks → upper-right (strong forwards + strong D). "
        "Upper-left = forward-heavy stacks; lower-right = D-heavy but weaker forwards."
    ),
    showarrow=False,
    font=dict(size=11),
)

# ----------------------------------------------------------
# Layout
# ----------------------------------------------------------
fig.update_layout(
    height=1600,
    width=1100,
    title_text="NHL Line Boom Stack Dashboard",
    showlegend=False,
    margin=dict(l=60, r=30, t=80, b=50),
)

fig.add_annotation(
    x=0,
    y=1.08,
    xref="paper",
    yref="paper",
    text=(
        "Legend/Key: Colors = teams. "
        "Row 1: elite lines → upper-right. "
        "Row 2: value D stacks → upper-left. "
        "Row 3: balanced F + D → upper-right."
    ),
    showarrow=False,
    font=dict(size=11),
)

# ----------------------------------------------------------
# Output HTML
# ----------------------------------------------------------
print(" Writing HTML report ->", OUTPUT_HTML)
fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
print(" Done.")
