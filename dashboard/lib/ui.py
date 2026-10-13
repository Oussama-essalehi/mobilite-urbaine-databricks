"""Charte graphique et composants partagés : palette, CSS, en-têtes, KPI, graphiques Plotly, tables."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Palette (ordre catégoriel fixe, validé daltonisme) -------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA = SERIES[0], SERIES[1], SERIES[2]
MUTED = "#898781"
GRID = "rgba(137,135,129,0.22)"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0, "#d03b3b"], [0.5, "#f0efec"], [1, "#2a78d6"]]

# Statuts : manque de vélos côté rouge, manque de bornes côté bleu, nominal en vert
AVAILABILITY = {
    "Vide": "#d03b3b",
    "Presque vide": "#ec835a",
    "Disponible": "#0ca30c",
    "Presque pleine": "#6da7ec",
    "Pleine": "#184f95",
    "Fermée": "#898781",
}
ACTION = {"REAPPROVISIONNER": "#d03b3b", "DELESTER": "#184f95", "OK": "#b5b3ab"}
ACTION_LABEL = {"REAPPROVISIONNER": "À réapprovisionner", "DELESTER": "À délester", "OK": "Équilibrée"}
DAY_TYPE = {"Semaine": BLUE, "Week-end": ORANGE}

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

CSS = """
<style>
:root { --accent:#2a78d6; --ink-2:#52514e; --muted:#898781; --hair:rgba(11,11,11,.10); }
.block-container { padding-top: 4.6rem; padding-bottom: 3rem; max-width: 1320px; }
h1, h2, h3 { letter-spacing: -0.01em; }
[data-testid="stMetric"] { padding: 14px 16px 12px; border-radius: 12px; }
[data-testid="stMetricLabel"] p { font-size: .82rem; color: var(--ink-2); font-weight: 500; }
[data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 650; }
.page-eyebrow { font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
                color: var(--accent); margin-bottom: .1rem; }
.page-title { font-size: 2.05rem; font-weight: 700; line-height: 1.15; margin: 0 0 .35rem; }
.page-sub { color: var(--ink-2); font-size: 1.02rem; max-width: 900px; margin-bottom: 1.1rem; }
.section { font-size: 1.12rem; font-weight: 650; margin: 1.4rem 0 .15rem; }
.section-sub { color: var(--muted); font-size: .88rem; margin-bottom: .55rem; }
.insight { border-left: 3px solid var(--accent); padding: .55rem .9rem; margin: .4rem 0 .9rem;
           background: rgba(42,120,214,.06); border-radius: 0 8px 8px 0; font-size: .95rem; }
.insight b { font-weight: 650; }
.tag { display:inline-block; padding: 3px 10px; margin: 0 6px 6px 0; border-radius: 999px;
       border: 1px solid var(--hair); font-size: .8rem; color: var(--ink-2); background: rgba(127,127,127,.06); }
.hero { padding: 28px 30px 22px; border-radius: 16px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, rgba(42,120,214,.12), rgba(27,175,122,.08) 60%, rgba(235,104,52,.06));
        border: 1px solid var(--hair); }
.hero h1 { font-size: 2.35rem; margin: 0 0 .4rem; line-height: 1.1; }
.hero p { font-size: 1.07rem; color: var(--ink-2); max-width: 860px; margin: 0 0 .9rem; }
.flow { display:grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 14px; margin: .4rem 0 1rem; }
@media (max-width: 1100px) { .flow { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); } }
.flow .step { position: relative; border: 1px solid var(--hair); border-radius: 12px;
              padding: 10px 12px; background: rgba(127,127,127,.04); }
.flow .step .k { font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
.flow .step .t { font-weight: 650; font-size: .95rem; margin: 2px 0 4px; }
.flow .step .d { font-size: .8rem; color: var(--ink-2); line-height: 1.35; }
.flow .step:not(:last-child)::after { content: "›"; position: absolute; right: -11px; top: 50%;
              transform: translateY(-50%); color: var(--muted); font-size: 1.1rem; }
.card { border: 1px solid var(--hair); border-radius: 12px; padding: 14px 16px; height: 100%; }
.card .t { font-weight: 650; margin-bottom: 4px; }
.card .d { font-size: .88rem; color: var(--ink-2); line-height: 1.45; }
.legend-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:baseline; }
.src-badge { font-size: .8rem; padding: 8px 10px; border-radius: 10px; border: 1px solid var(--hair); line-height: 1.4; }
.src-badge b { display:block; font-size: .85rem; }
footer { visibility: hidden; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="page-eyebrow">{eyebrow}</div><div class="page-title">{title}</div>'
                f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def section(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="section">{title}</div>' + (f'<div class="section-sub">{subtitle}</div>' if subtitle else ""),
                unsafe_allow_html=True)


def insight(html: str) -> None:
    st.markdown(f'<div class="insight">{html}</div>', unsafe_allow_html=True)


def kpis(items: list[tuple]) -> None:
    """items : (libellé, valeur, delta | None, aide | None[, delta_color])"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, delta, help_ = (list(item) + [None] * 4)[:4]
        delta_color = item[4] if len(item) > 4 else "normal"
        col.metric(label, value, delta, help=help_, border=True, delta_color=delta_color)


def missing(what: str) -> None:
    st.info(f"{what} — table non disponible sur cette source (le job correspondant ne l'a pas encore produite).",
            icon=":material/hourglass_empty:")


# --- Plotly --------------------------------------------------------------------------------------------

def style(fig: go.Figure, height: int = 360, legend: bool = True, hover: str = "x unified") -> go.Figure:
    has_title = bool(fig.layout.title.text)
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=44 if has_title else 36 if legend else 12, b=8),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=13),
        hovermode=hover, showlegend=legend, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text="",
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="", title_font=dict(size=12, color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font=dict(size=12, color=MUTED))
    return fig


def plot(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)


TOULOUSE = {"lat": 43.6045, "lon": 1.4440}
CLUSTER_SLOTS = {"Résidentielle": 0, "Pôle d'activité": 1, "Équilibrée": 2, "Souvent vide": 7, "Souvent pleine": 6}


def cluster_colors(labels) -> dict:
    """Couleur fixe par type de segment (jamais par rang), libellés dédoublonnés sur les créneaux libres."""
    out, used = {}, set()
    for lab in sorted(set(labels)):
        slot = next((s for k, s in CLUSTER_SLOTS.items() if lab.startswith(k)), None)
        if slot is None or slot in used:
            slot = next(i for i in range(len(SERIES)) if i not in used and i not in CLUSTER_SLOTS.values())
        used.add(slot)
        out[lab] = SERIES[slot]
    return out


def map_style() -> str:
    return "carto-darkmatter" if st.context.theme.type == "dark" else "carto-positron"


def station_map(df: pd.DataFrame, color: str, color_map: dict, hover: dict, size: str | None = None,
                height: int = 540, zoom: float = 11.4, size_max: int = 13) -> go.Figure:
    import plotly.express as px

    fig = px.scatter_map(
        df, lat="lat", lon="lon", color=color, color_discrete_map=color_map,
        category_orders={color: [c for c in color_map if c in set(df[color])]},
        size=size, size_max=size_max, hover_name="station_label", hover_data=hover,
        zoom=zoom, center=TOULOUSE, map_style=map_style(), height=height,
    )
    if size is None:
        fig.update_traces(marker=dict(size=9))
    fig.update_traces(marker=dict(opacity=0.9))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(
        orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.01, title_text="",
        bgcolor="rgba(255,255,255,0.85)" if map_style() == "carto-positron" else "rgba(26,26,25,0.85)"))
    return fig


def data_table(df: pd.DataFrame, label: str = "Voir les données", **kwargs) -> None:
    with st.expander(label, icon=":material/table:"):
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)


def fmt_int(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def fmt_pct(x: float, digits: int = 0, signed: bool = False) -> str:
    if round(x * 100, digits) == 0:
        x, signed = 0.0, False
    s = f"{x * 100:+.{digits}f}" if signed else f"{x * 100:.{digits}f}"
    return s.replace(".", ",") + " %"


def fmt_num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")
