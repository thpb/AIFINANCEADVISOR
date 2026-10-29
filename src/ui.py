"""Look and feel: CSS, colours, Plotly chart functions and small HTML helpers."""
import html
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from budgets import money

CATEGORIES = ["Bills", "Dining", "Entertainment", "Groceries", "Other", "Shopping", "Subscriptions", "Transport"]
PALETTE_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
FALLBACK = "#8a8a85"
STATUS = {"ok": ("#0ca30c", "✓"), "warning": ("#fab219", "!"), "over": ("#d03b3b", "✕")}
PLOT_CONFIG = {"displayModeBar": False, "scrollZoom": False}

CSS = """
<style>
.block-container {max-width:1180px; padding-top:4rem; padding-bottom:6rem}
.st-key-quickadd input {font-size:1.1rem; padding:0.7rem 0.8rem}
.app-title {font-size:0.95rem; font-weight:600; letter-spacing:0.02em; opacity:0.75; margin-bottom:0.4rem}
.hero {font-size:2.1rem; font-weight:650; line-height:1.2; font-variant-numeric:tabular-nums; margin:0.2rem 0 0.4rem}
.hero-note {font-size:1rem; opacity:0.8; margin-bottom:1.2rem; line-height:1.5}
.stat {padding:0.2rem 0}
.stat-label {font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; opacity:0.65}
.stat-value {font-size:1.45rem; font-weight:600; font-variant-numeric:tabular-nums; line-height:1.3}
.stat-note {font-size:0.85rem; opacity:0.7}
.budget-row {margin:0.2rem 0 1rem}
.budget-top {display:flex; justify-content:space-between; gap:1rem; font-variant-numeric:tabular-nums}
.budget-name {font-weight:600}
.budget-bar {height:8px; border-radius:4px; background:rgba(128,128,128,0.18); margin:0.35rem 0; overflow:hidden}
.budget-fill {height:100%; border-radius:4px}
.budget-state {font-size:0.85rem}
@media (max-width:640px) {
  .block-container {padding-left:1rem; padding-right:1rem; padding-top:3.5rem}
  .hero {font-size:1.6rem}
  .stat-value {font-size:1.2rem}
}
</style>
"""


def inject_css():
    st.html(CSS)


def is_dark():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def page_bg():
    return "#0e1117" if is_dark() else "#ffffff"


def ink():
    return "#e8e8e3" if is_dark() else "#3d3d3a"


def category_color(category):
    """Fixed slot per category (alphabetical), so a category has the same colour in every chart."""
    palette = PALETTE_DARK if is_dark() else PALETTE_LIGHT
    if category in CATEGORIES:
        return palette[CATEGORIES.index(category)]
    return FALLBACK


# ---------- text safety ----------

def esc(text):
    """Escape $ so Streamlit doesn't read text between two dollar signs as a maths formula."""
    return str(text).replace("\\$", "$").replace("$", "\\$")


def plain_dollars(text):
    text = text.replace("`", "")
    text = text.replace("\u202f", " ").replace("\u00a0", " ")
    text = re.sub(r"\\[()\[\]]", "", text)                              # \( \) \[ \]
    text = re.sub(r"\\(?:text|mathrm|textbf)\{([^}]*)\}", r"\1", text)  # \text{...}
    text = text.replace("{,}", ",").replace("$$", "$")
    return text.replace("\\$", "$").replace("$", "\\$")


def no_headings(text):
    """The prompt says no headings, but small models still write '### Tips'. Show them as bold text instead."""
    return re.sub(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", r"**\1**", text, flags=re.M)


# ---------- HTML helpers (st.html: no markdown, so $ is safe) ----------

def app_title():
    st.html('<div class="app-title">💳 Expense Categorizer</div>')


def hero(text, note=""):
    note_html = f'<div class="hero-note">{html.escape(note)}</div>' if note else ""
    st.html(f'<div class="hero">{html.escape(text)}</div>{note_html}')


def stat(label, value, note=""):
    st.html(f'<div class="stat"><div class="stat-label">{html.escape(label)}</div>'
            f'<div class="stat-value">{html.escape(value)}</div>'
            f'<div class="stat-note">{html.escape(note)}</div></div>')


def budget_row(r):
    color, icon = STATUS[r["state"]]
    if r["state"] == "ok":
        word = f"On track, {money(r['remaining'])} left"
    elif r["state"] == "warning":
        word = f"Close to the limit, {money(r['remaining'])} left"
    else:
        word = f"Over by {money(-r['remaining'])}"
    width = min(r["pct"], 1) * 100
    st.html(
        f'<div class="budget-row"><div class="budget-top"><span class="budget-name">{html.escape(r["category"])}'
        f'</span><span>{money(r["spent"])} of {money(r["limit"])}</span></div>'
        f'<div class="budget-bar"><div class="budget-fill" style="width:{width:.1f}%;background:{color}"></div></div>'
        f'<div class="budget-state" style="color:{color}"><b>{icon}</b> {html.escape(word)}</div></div>')


# ---------- charts ----------

def _base(fig, height, legend=False):
    grid = "rgba(128,128,128,0.15)"
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", dragmode=False, showlegend=legend, bargap=0.45,
        font=dict(color=ink(), size=13), hoverlabel=dict(font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    )
    fig.update_xaxes(gridcolor=grid, zeroline=False, fixedrange=True, automargin=True)
    fig.update_yaxes(gridcolor=grid, zeroline=False, fixedrange=True, automargin=True, ticksuffix="  ")
    return fig


def _rounded(fig):
    try:
        fig.update_layout(barcornerradius=4)
    except Exception:       # older Plotly has no corner radius
        pass
    return fig


def show(fig, alt=""):
    st.plotly_chart(fig, config=PLOT_CONFIG, theme=None, alt=alt or None)


def hbar(series, by_category=True):
    """Horizontal sorted bars with direct $ labels (used instead of pie charts)."""
    s = series.sort_values()
    colors = [category_color(c) if by_category else category_color("Bills") for c in s.index]
    fig = go.Figure(go.Bar(
        x=s.values, y=[str(i) for i in s.index], orientation="h", marker=dict(color=colors),
        text=[money(v) for v in s.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{text}<extra></extra>"))
    _base(fig, height=max(160, 44 * len(s) + 30))
    fig.update_xaxes(showgrid=False, showticklabels=False, range=[0, float(s.max()) * 1.25 if len(s) else 1])
    return _rounded(fig)


def month_chart(chart_df, income_by_month=None, highlight=None):
    """Stacked monthly bars per category + optional income line on the same dollar axis."""
    months = (chart_df[["month", "month_str"]].drop_duplicates().sort_values("month"))
    labels = list(months["month_str"])
    fig = go.Figure()
    cats = sorted(chart_df["category"].unique())
    for cat in cats:
        rows = chart_df[chart_df["category"] == cat].set_index("month")
        values = [float(rows["amount"].get(m, 0.0)) for m in months["month"]]
        opacity = [1.0 if (highlight is None or m == highlight) else 0.45 for m in months["month"]]
        fig.add_bar(x=labels, y=values, name=cat,
                    marker=dict(color=category_color(cat), opacity=opacity, line=dict(color=page_bg(), width=2)),
                    hovertemplate=f"{cat}: %{{y:$,.2f}}<extra></extra>")
    series = 1 + (1 if income_by_month is not None and len(income_by_month) else 0)
    if income_by_month is not None and len(income_by_month):
        y = [float(income_by_month.get(m, 0.0)) for m in months["month"]]
        fig.add_scatter(x=labels, y=y, name="Income", mode="lines+markers",
                        line=dict(color=ink(), width=2), marker=dict(size=8, color=ink()),
                        hovertemplate="Income: %{y:$,.2f}<extra></extra>")
    _base(fig, height=360, legend=len(cats) + series - 1 >= 2)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_xaxes(showgrid=False)
    return _rounded(fig)


def forecast_chart(fc):
    """Solid bar = spent so far, faded bar = expected rest, red tick = budget."""
    rows = sorted([r for r in fc["categories"] if r["projected"] > 0 or r["limit"]], key=lambda r: r["projected"])
    names = [r["category"] for r in rows]
    colors = [category_color(n) for n in names]
    fig = go.Figure()
    fig.add_bar(y=names, x=[r["spent"] for r in rows], orientation="h", name="Spent so far",
                marker=dict(color=colors), hovertemplate="%{y}: %{x:$,.2f} so far<extra></extra>")
    fig.add_bar(y=names, x=[max(r["projected"] - r["spent"], 0) for r in rows], orientation="h",
                name="Expected rest", marker=dict(color=colors, opacity=0.35),
                text=[money(round(r["projected"])) for r in rows], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: about %{text} by month end<extra></extra>")
    budget_rows = [r for r in rows if r["limit"]]
    if budget_rows:
        fig.add_scatter(y=[r["category"] for r in budget_rows], x=[r["limit"] for r in budget_rows], mode="markers",
                        name="Budget", marker=dict(symbol="line-ns", size=26, line=dict(width=3, color="#d03b3b")),
                        hovertemplate="%{y} budget: %{x:$,.0f}<extra></extra>")
    top = max([r["projected"] for r in rows] + [r["limit"] or 0 for r in rows] + [1])
    _base(fig, height=max(180, 46 * len(rows) + 30))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(showgrid=False, showticklabels=False, range=[0, top * 1.22])
    return _rounded(fig)


def month_rows(df):
    """Monthly totals per category in the shape month_chart wants."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], format="mixed")
    d["month"] = d["date"].dt.strftime("%Y-%m")
    col = "predicted_category" if "predicted_category" in d.columns else "category"
    g = d.groupby(["month", col])["amount"].sum().round(2).reset_index()
    g.columns = ["month", "category", "amount"]
    g["month_str"] = g["month"].map(lambda m: pd.Period(m, freq="M").strftime("%b %Y"))
    return g
