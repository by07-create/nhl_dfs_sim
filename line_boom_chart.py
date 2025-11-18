import pandas as pd
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
PROJECT_DIR = Path(r"C:\Users\b-yan\OneDrive\Documents\Bo - Python Apps\NHL Simulator")
INPUT_FILE = PROJECT_DIR / "nhl_line_boom_model.csv"
OUTPUT_HTML = PROJECT_DIR / "nhl_line_boom_charts.html"

print(f" Loading line model -> {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"   Loaded {len(df)} line rows.")

# ----------------------------------------------------------
# Ensure numeric types (safety)
# ----------------------------------------------------------
numeric_cols = [
    "line_score", "line_boom_pct", "line_sim_mean", "line_sim_p90",
    "line_salary_total", "line_salary_mean", "line_value_score",
    "goalie_mult", "line_proj_total", "line_xg_pg",
    "fwd_score", "fwd_boom_pct", "fwd_xg_pg",
    "matchup_softness"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------------------------------------------
# Simple team color mapping
# ----------------------------------------------------------
teams = sorted(df["TEAM"].astype(str).unique())
palette = px.colors.qualitative.Plotly
color_map = {
    team: palette[i % len(palette)]
    for i, team in enumerate(teams)
}

def team_color(series):
    return series.astype(str).map(color_map)

# Marker size based on salary (for chart 1)
if "line_salary_total" in df.columns:
    sal = df["line_salary_total"].fillna(0)
    if sal.max() > sal.min():
        size_scaled = 8 + (sal - sal.min()) / (sal.max() - sal.min()) * 20
    else:
        size_scaled = pd.Series(12, index=df.index)
else:
    size_scaled = pd.Series(12, index=df.index)

# ----------------------------------------------------------
# Create subplots: 5 rows x 1 column
# ----------------------------------------------------------
fig = make_subplots(
    rows=5,
    cols=1,
    shared_xaxes=False,
    vertical_spacing=0.08,
    subplot_titles=(
        "1. Line Score vs Boom %",
        "2. Average Salary vs Value Score",
        "3. Simulated Mean vs P90",
        "4. Matchup Softness vs Line Score",
        "5. Forward xG per Game vs Forward Boom %"
    )
)

# ----------------------------------------------------------
# Row 1: Line Score vs Boom % (bubble = salary)
# ----------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=df["line_score"],
        y=df["line_boom_pct"],
        mode="markers",
        marker=dict(
            size=size_scaled,
            color=team_color(df["TEAM"]),
            sizemode="diameter",
            opacity=0.8,
            line=dict(width=0.5, color="black")
        ),
        text=df["TEAM"] + " " + df["env_key"].astype(str),
        hovertemplate=(
            "TEAM: %{text}<br>"
            "Line Score: %{x:.2f}<br>"
            "Boom %: %{y:.2f}<br>"
            "Salary Total: %{customdata[0]}<extra></extra>"
        ),
        customdata=df[["line_salary_total"]].values,
        name="Line Score vs Boom %"
    ),
    row=1,
    col=1
)

fig.update_yaxes(title_text="Boom %", row=1, col=1)
fig.update_xaxes(title_text="Line Score", row=1, col=1)

# Explanation annotation for row 1
fig.add_annotation(
    row=1, col=1,
    x=0, y=1.12,
    xref="x domain", yref="y domain",
    text="Best GPP lines -> upper-right (high Line Score, high Boom%). Larger bubbles = more salary.",
    showarrow=False,
    font=dict(size=11)
)

# ----------------------------------------------------------
# Row 2: Salary Mean vs Value Score
# ----------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=df["line_salary_mean"],
        y=df["line_value_score"],
        mode="markers",
        marker=dict(
            size=10,
            color=team_color(df["TEAM"]),
            opacity=0.85,
            line=dict(width=0.5, color="black")
        ),
        text=df["TEAM"] + " " + df["env_key"].astype(str),
        hovertemplate=(
            "TEAM: %{text}<br>"
            "Avg Salary: %{x:.0f}<br>"
            "Value Score: %{y:.3f}<extra></extra>"
        ),
        name="Salary vs Value"
    ),
    row=2,
    col=1
)

fig.update_yaxes(title_text="Value Score", row=2, col=1)
fig.update_xaxes(title_text="Average Salary per Skater", row=2, col=1)

fig.add_annotation(
    row=2, col=1,
    x=0, y=1.12,
    xref="x domain", yref="y domain",
    text="Best value stacks -> upper-left (cheap but strong Value Score).",
    showarrow=False,
    font=dict(size=11)
)

# ----------------------------------------------------------
# Row 3: Sim mean vs P90 (ceiling)
# ----------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=df["line_sim_mean"],
        y=df["line_sim_p90"],
        mode="markers",
        marker=dict(
            size=10,
            color=df["line_boom_pct"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Boom %"),
            opacity=0.85
        ),
        text=df["TEAM"] + " " + df["env_key"].astype(str),
        hovertemplate=(
            "TEAM: %{text}<br>"
            "Sim Mean: %{x:.2f}<br>"
            "P90: %{y:.2f}<br>"
            "Boom %: %{marker.color:.2f}<extra></extra>"
        ),
        name="Mean vs P90"
    ),
    row=3,
    col=1
)

fig.update_yaxes(title_text="P90 (Ceiling)", row=3, col=1)
fig.update_xaxes(title_text="Simulated Mean", row=3, col=1)

fig.add_annotation(
    row=3, col=1,
    x=0, y=1.12,
    xref="x domain", yref="y domain",
    text="Best ceiling lines -> upper-right (high Mean & high P90). Darker color = higher Boom%.",
    showarrow=False,
    font=dict(size=11)
)

# ----------------------------------------------------------
# Row 4: Matchup Softness vs Line Score
# ----------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=df["matchup_softness"],
        y=df["line_score"],
        mode="markers",
        marker=dict(
            size=10,
            color=team_color(df["TEAM"]),
            opacity=0.85,
            line=dict(width=0.5, color="black")
        ),
        text=df["TEAM"] + " " + df["env_key"].astype(str),
        hovertemplate=(
            "TEAM: %{text}<br>"
            "Matchup Softness: %{x:.3f}<br>"
            "Line Score: %{y:.2f}<extra></extra>"
        ),
        name="Matchup vs Strength"
    ),
    row=4,
    col=1
)

fig.update_yaxes(title_text="Line Score", row=4, col=1)
fig.update_xaxes(title_text="Matchup Softness (softer D -> right)", row=4, col=1)

fig.add_annotation(
    row=4, col=1,
    x=0, y=1.12,
    xref="x domain", yref="y domain",
    text="Best combo -> top-right (strong line in soft matchup). Leverage -> top-left (strong line vs tough D).",
    showarrow=False,
    font=dict(size=11)
)

# ----------------------------------------------------------
# Row 5: Forward xG per Game vs Forward Boom %
# ----------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=df["fwd_xg_pg"],
        y=df["fwd_boom_pct"],
        mode="markers",
        marker=dict(
            size=10,
            color=team_color(df["TEAM"]),
            opacity=0.85,
            line=dict(width=0.5, color="black")
        ),
        text=df["TEAM"] + " " + df["env_key"].astype(str),
        hovertemplate=(
            "TEAM: %{text}<br>"
            "Fwd xG per Game: %{x:.3f}<br>"
            "Fwd Boom %: %{y:.3f}<extra></extra>"
        ),
        name="Forward xG vs Boom"
    ),
    row=5,
    col=1
)

fig.update_yaxes(title_text="Forward Boom %", row=5, col=1)
fig.update_xaxes(title_text="Forward xG per Game", row=5, col=1)

fig.add_annotation(
    row=5, col=1,
    x=0, y=1.12,
    xref="x domain", yref="y domain",
    text="Best forward stacks -> upper-right (high xG and high Boom%).",
    showarrow=False,
    font=dict(size=11)
)

# ----------------------------------------------------------
# Layout tweaks
# ----------------------------------------------------------
fig.update_layout(
    height=1800,
    width=1100,
    title_text="NHL Line Boom Model - Slate Overview",
    showlegend=False,
    margin=dict(l=60, r=30, t=80, b=50)
)

# ----------------------------------------------------------
# Show interactive figure (single window)
# ----------------------------------------------------------
print(" Showing combined interactive figure (all 5 charts stacked)")
fig.show()

# ----------------------------------------------------------
# Write HTML report
# ----------------------------------------------------------
print(f" Writing HTML report -> {OUTPUT_HTML}")
fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
print(" Done.")
