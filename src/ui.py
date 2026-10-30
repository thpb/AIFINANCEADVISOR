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

# Option C colours. The category palette and budget status colours above stay as they are (colour-blind safe).
DARK = "#13261F"
AMBER = "#E3A33C"
MUTED = "#4F5B55"
CREAM = "#F5F3EC"
LINE = "#DDE3DA"
FONT = "'IBM Plex Sans', sans-serif"
HEADING_FONT = "'Space Grotesk', 'IBM Plex Sans', sans-serif"

CSS = """
<style>
/* ---- page frame: no sidebar or Streamlit header, content 1180px wide ---- */
header[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {display:none !important}
.block-container, [data-testid="stMainBlockContainer"] {max-width:1340px; padding-top:0 !important; padding-bottom:5rem}
h1, h2, h3, .tile-title, .hero-big {font-family:'Space Grotesk', 'IBM Plex Sans', sans-serif; letter-spacing:-0.01em}

/* full-width coloured bands behind normal-width content */
.st-key-topbar, .st-key-hero {background:#13261F; box-shadow:0 0 0 100vmax #13261F; clip-path:inset(0 -100vmax)}
.st-key-hero {clip-path:inset(-1.2rem -100vmax 0 -100vmax); padding:1.4rem 0 2rem; margin-bottom:0.6rem}
.st-key-strip {background:#E8EDE6; box-shadow:0 0 0 100vmax #E8EDE6; clip-path:inset(0 -100vmax);
  padding:0.35rem 0; border-bottom:1px solid #DDE3DA; margin-top:-1rem; margin-bottom:0.6rem}

/* ---- top bar ---- */
.st-key-topbar {padding:0.75rem 0}
.brand {display:flex; align-items:center; gap:10px; color:#F5F3EC; font-family:'Space Grotesk', sans-serif;
  font-weight:700; font-size:19px; white-space:nowrap}
.brand-logo {width:30px; height:30px; border-radius:7px; background:#F5F3EC; display:flex; align-items:flex-end;
  justify-content:center; gap:3px; padding:0 0 7px}
.brand-logo i {display:block; width:4px; border-radius:1px}
.st-key-nav [data-testid="stButtonGroup"] > div {gap:4px; flex-wrap:wrap}
.st-key-nav button {background:transparent !important; border:none !important; box-shadow:none !important;
  border-radius:999px !important; padding:0.35rem 0.95rem !important; min-height:2.3rem}
.st-key-nav button p {color:#E8EDE6; font-size:15px; font-weight:500}
.st-key-nav button:hover p {color:#FFFFFF}
.st-key-nav button[aria-checked="true"] {background:#E3A33C !important}
.st-key-nav button[aria-checked="true"] p {color:#13261F !important; font-weight:600}
/* the quick-add box takes the free space in the bar (its flex wrapper is the element without the key) */
.st-key-topbar > div:has(.st-key-quickadd) {flex:1 1 260px !important; min-width:220px; max-width:480px;
  margin-left:auto}
.st-key-quickadd {width:100% !important}
.st-key-quickadd [data-testid="stForm"] {padding:0}
.st-key-quickadd [data-testid="stTextInputRootElement"] {background:#1C3229 !important;
  border:1px solid #2E4A3E !important; border-radius:12px}
.st-key-quickadd input {color:#F5F3EC !important; -webkit-text-fill-color:#F5F3EC; font-size:15px}
.st-key-quickadd input::placeholder {color:#9FB0A7; -webkit-text-fill-color:#9FB0A7}
.st-key-quickadd button, [class*="st-key-amber_"] button {background:#E3A33C !important; border:none !important;
  border-radius:10px !important}
.st-key-quickadd button p, [class*="st-key-amber_"] button p {color:#13261F !important; font-weight:600}
.st-key-settings button {background:transparent !important; border:1px solid #3A5248 !important; border-radius:999px !important}
.st-key-settings button p, .st-key-settings button span {color:#E8EDE6 !important}

/* ---- Overview hero band ---- */
.hero-month {color:#C9D3CC; font-size:15px; margin-bottom:2px}
.hero-big {font-size:44px; font-weight:700; color:#F5F3EC; line-height:1.1; font-variant-numeric:tabular-nums}
.hero-big .amber {color:#E3A33C}
.hero-notes {color:#AEBBB3; font-size:14px; margin-top:10px; line-height:1.5; max-width:640px}
.progress-top {display:flex; justify-content:space-between; gap:12px; color:#D5DDD8; font-size:13px}
.progress {position:relative; height:12px; background:#2E4A3E; border-radius:999px; margin:8px 0 10px}
.progress .fill {height:100%; background:#E3A33C; border-radius:999px}
.progress .today {position:absolute; top:-5px; width:2px; height:22px; background:#FFFFFF; border-radius:1px}
.progress-note {color:#D5DDD8; font-size:13px; line-height:1.45}
.st-key-hero label p {color:#C9D3CC !important}
.st-key-hero [data-testid="stSelectbox"] [role="group"] {background:#1C3229 !important; border:1px solid #2E4A3E}
.st-key-hero [data-testid="stSelectbox"] input {color:#F5F3EC !important; -webkit-text-fill-color:#F5F3EC}
.st-key-hero [data-testid="stSelectbox"] svg {fill:#F5F3EC; color:#F5F3EC}
.st-key-hero [data-testid="stSelectbox"] {max-width:240px; margin-left:auto}

/* ---- bento tiles ---- */
[class*="st-key-tile_"] {background:#FFFFFF; border:1px solid #DDE3DA; border-radius:18px; padding:20px 22px 18px}
.st-key-tile_pace {background:#E3A33C; border-color:#E3A33C}
.st-key-tile_ask {background:#13261F; border-color:#13261F}
.st-key-tile_ask p, .st-key-tile_ask div {color:#E8EDE6}
.tile-head {display:flex; justify-content:space-between; align-items:baseline; gap:10px}
.tile-title {font-size:20px; font-weight:600; color:#13261F}
.tile-side {color:#4F5B55; font-size:13px}
.tile-label {font-size:14px; font-weight:600; color:#13261F}
.pace-big {font-family:'Space Grotesk', sans-serif; font-size:44px; font-weight:700; color:#13261F; line-height:1.1;
  margin:14px 0 12px; font-variant-numeric:tabular-nums}
.pace-text {font-size:14px; line-height:1.5; color:#13261F}
.ask-label {color:#E3A33C !important; font-size:14px; font-weight:600}
.ask-quote {font-family:'Space Grotesk', sans-serif; font-size:21px; font-weight:700; color:#FFFFFF !important;
  line-height:1.3; margin:14px 0 12px}
.ask-text {font-size:14px; line-height:1.5; color:#C9D3CC !important}
[class*="st-key-link_"] button {background:transparent !important; border:none !important; padding:0 !important;
  min-height:0 !important}
[class*="st-key-link_"] button p {color:#1F5C45; font-weight:600; text-decoration:underline; text-underline-offset:3px}
.st-key-tile_pace [class*="st-key-link_"] button p {color:#13261F}
.recent {width:100%; border-collapse:collapse; font-size:14px; margin-top:8px}
.recent td {padding:10px 4px; border-bottom:1px solid #E8EDE6; vertical-align:middle; white-space:nowrap}
.recent tr:last-child td {border-bottom:none}
.recent .d {color:#4F5B55; white-space:nowrap; width:62px}
.recent .m {font-weight:600; max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.recent .a {text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums}
.dot {display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; vertical-align:0}
.badge {display:inline-block; margin-left:6px; padding:1px 7px; border-radius:999px; background:#FCEFD6;
  color:#6B4A0E; font-size:12px}
.budget-row {padding:12px 0; border-bottom:1px solid #E8EDE6}
.budget-row:last-of-type {border-bottom:none}
.budget-top {display:flex; justify-content:space-between; gap:1rem; font-variant-numeric:tabular-nums}
.budget-name {font-weight:600}
.budget-bar {height:8px; border-radius:4px; background:#E3E7E1; margin:0.4rem 0; overflow:hidden}
.budget-fill {height:100%; border-radius:4px}
.budget-state {font-size:13px; color:#4F5B55}

/* ---- quick add confirm step ---- */
.st-key-confirm_box {background:#FFFFFF; border-left:4px solid #E3A33C !important; border-radius:14px}

/* ---- older page headers (Plan ahead) ---- */
.hero {font-family:'Space Grotesk', sans-serif; font-size:2.1rem; font-weight:700; line-height:1.2;
  font-variant-numeric:tabular-nums; margin:0.2rem 0 0.4rem}
.hero-note {font-size:1rem; color:#4F5B55; margin-bottom:1.2rem; line-height:1.5}

/* ---- AI advisor ---- */
.strip {display:flex; flex-wrap:wrap; gap:6px 18px; font-size:14px; color:#4F5B55; align-items:center}
.strip b {color:#13261F}
.st-key-strip button p {color:#1F5C45 !important; font-weight:600; text-decoration:underline; text-underline-offset:3px}
.st-key-strip button {background:transparent !important; border:none !important; padding:0 4px !important;
  min-height:0 !important}
.st-key-chat {max-width:860px; margin:0.4rem auto 0; width:100%}
.st-key-chat [data-testid="stMarkdownContainer"] p, .st-key-chat li {font-size:16px; line-height:1.7}
.user-row {display:flex; justify-content:flex-end}
.user-bubble {background:#13261F; color:#F5F3EC; border-radius:14px; padding:10px 18px; max-width:80%;
  font-weight:500; font-size:16px; line-height:1.5}
.plan-grid {display:grid; grid-template-columns:repeat(auto-fill, minmax(165px, 1fr)); gap:12px; margin:4px 0}
.plan-tile {background:#FFFFFF; border:1px solid #DDE3DA; border-radius:14px; padding:14px 18px}
.plan-cat {font-size:13px; color:#4F5B55}
.plan-cut {font-family:'Space Grotesk', sans-serif; font-size:24px; font-weight:700; margin:4px 0 8px;
  font-variant-numeric:tabular-nums}
.plan-bar {height:6px; border-radius:3px; background:#E3E7E1; overflow:hidden}
.plan-bar div {height:100%; border-radius:3px}
.plan-total {font-size:14px; margin-top:4px}
.plan-total .ok {color:#1F5C45; font-weight:600; margin-right:14px}
.plan-total .muted {color:#4F5B55}
.st-key-suggest button {border-radius:999px !important; background:#FFFFFF !important; border:1px solid #CBD3CC !important}
.st-key-suggest button p {font-weight:500}
[data-testid="stChatInput"], [data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea {
  background:#FFFFFF !important}
[data-testid="stChatInput"] {border:2px solid #13261F !important; border-radius:14px !important}
[data-testid="stChatInputSubmitButton"] {background:#E3A33C !important; border-radius:10px !important}
[data-testid="stChatInputSubmitButton"] svg {fill:#13261F !important; color:#13261F !important}

@media (max-width:640px) {
  .block-container, [data-testid="stMainBlockContainer"] {padding-left:1rem; padding-right:1rem}
  .hero-big {font-size:32px}
  .hero {font-size:1.6rem}
  .pace-big {font-size:36px}
  .brand {font-size:17px}
  .recent .m {max-width:110px}
  [class*="st-key-tile_"] {padding:16px 16px 14px}
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
    return "#e8e8e3" if is_dark() else DARK


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

def brand():
    """Small logo (three bars) and the app name for the top bar."""
    bars = "".join(f'<i style="height:{h}px;background:{c}"></i>' for h, c in [(8, AMBER), (14, DARK), (10, AMBER)])
    st.html(f'<div class="brand"><span class="brand-logo">{bars}</span>Expense Categorizer</div>')


def hero(text, note=""):
    note_html = f'<div class="hero-note">{html.escape(note)}</div>' if note else ""
    st.html(f'<div class="hero">{html.escape(text)}</div>{note_html}')


def hero_headline(month_text, spent_text, saved_text, notes):
    """Overview hero: month label, "$1,499 spent, $2,801 saved" (saved part amber) and a muted line of notes."""
    saved = f', <span class="amber">{html.escape(saved_text)}</span>' if saved_text else ""
    notes_html = f'<div class="hero-notes">{html.escape(notes)}</div>' if notes else ""
    st.html(f'<div class="hero-month">{html.escape(month_text)}</div>'
            f'<div class="hero-big">{html.escape(spent_text)}{saved}</div>{notes_html}')


def month_progress(left_label, right_label, fill, today_mark, note):
    """Amber bar = share spent, white line = how far through the month today is. fill/today_mark are 0-1 or None."""
    bar = ""
    if fill is not None:
        line = f'<div class="today" style="left:calc({today_mark * 100:.1f}% - 1px)"></div>' \
            if today_mark is not None else ""
        bar = (f'<div class="progress-top"><span>{html.escape(left_label)}</span>'
               f'<span>{html.escape(right_label)}</span></div>'
               f'<div class="progress"><div class="fill" style="width:{min(fill, 1) * 100:.1f}%"></div>{line}</div>')
    st.html(f'{bar}<div class="progress-note">{html.escape(note)}</div>')


def tile_head(title, side=""):
    side_html = f'<span class="tile-side">{html.escape(side)}</span>' if side else ""
    st.html(f'<div class="tile-head"><span class="tile-title">{html.escape(title)}</span>{side_html}</div>')


def pace_tile(label, big, text):
    st.html(f'<div class="tile-label">{html.escape(label)}</div><div class="pace-big">{html.escape(big)}</div>'
            f'<div class="pace-text">{html.escape(text)}</div>')


def ask_tile(question):
    st.html(f'<div class="ask-label">Ask the advisor</div><div class="ask-quote">“{html.escape(question)}”</div>'
            '<div class="ask-text">Get a plan worked out by the app, with tips from your local AI.</div>')


def recent_rows(rows):
    """rows: (date text, merchant, category, amount, needs a check) tuples."""
    body = "".join(
        f'<tr><td class="d">{html.escape(d)}</td><td class="m" title="{html.escape(m)}">{html.escape(m)}</td>'
        f'<td><span class="dot" style="background:{category_color(c)}"></span>{html.escape(c)}'
        f'{"<span class=badge>check</span>" if check else ""}</td><td class="a">{money(a)}</td></tr>'
        for d, m, c, a, check in rows)
    st.html(f'<table class="recent">{body}</table>')


def user_bubble(text):
    """The user's question as a dark bubble on the right (HTML, so $ amounts are always plain text)."""
    st.html(f'<div class="user-row"><div class="user-bubble">{html.escape(text)}</div></div>')


def plan_tiles(rows):
    """One tile per category: the cut (big) and a small bar sized against the biggest cut."""
    top = max((r["cut"] for r in rows), default=0) or 1
    tiles = "".join(
        f'<div class="plan-tile"><div class="plan-cat">{html.escape(r["category"])}</div>'
        f'<div class="plan-cut">-${r["cut"]:,.2f}</div><div class="plan-bar"><div style="width:{r["cut"] / top * 100:.0f}%;'
        f'background:{category_color(r["category"])}"></div></div></div>' for r in rows)
    st.html(f'<div class="plan-grid">{tiles}</div>')


def plan_total(total, note):
    st.html(f'<div class="plan-total"><span class="ok">✓ Total ${total:,.2f}, calculated by the app</span>'
            f'<span class="muted">{html.escape(note)}</span></div>')


def budget_row(r):
    """Status colour on the bar and icon, always with a word (never colour alone)."""
    color, icon = STATUS[r["state"]]
    if r["state"] == "ok":
        word = f"On track, {money(r['remaining'])} left"
    elif r["state"] == "warning":
        word = f"Close to the limit, {money(r['remaining'])} left"
    else:
        word = f"Over by {money(-r['remaining'])}"
    width = min(r["pct"], 1) * 100
    # The amber warning colour is too light for text on white, so the words stay dark; the icon keeps the colour.
    word_color = STATUS["over"][0] if r["state"] == "over" else MUTED
    st.html(
        f'<div class="budget-row"><div class="budget-top"><span class="budget-name">{html.escape(r["category"])}'
        f'</span><span>{r["pct"]:.0%}</span></div>'
        f'<div class="budget-bar"><div class="budget-fill" style="width:{width:.1f}%;background:{color}"></div></div>'
        f'<div class="budget-state" style="color:{word_color}" title="{money(r["spent"])} of {money(r["limit"])}">'
        f'<b style="color:{color}">{icon}</b> {html.escape(word)}</div></div>')


# ---------- charts ----------

def _base(fig, height, legend=False):
    """The one shared chart style: app fonts, transparent background, light grid, the same margins."""
    grid = "rgba(19,38,31,0.08)"
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", dragmode=False, showlegend=legend, bargap=0.45,
        font=dict(family=FONT, color=ink(), size=13),
        hoverlabel=dict(font=dict(family=FONT, size=13), bgcolor="#FFFFFF", bordercolor=LINE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None, font=dict(family=FONT)),
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
    fig.update_xaxes(showgrid=False, showticklabels=False, range=[0, float(s.max()) * 1.4 if len(s) else 1])
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
