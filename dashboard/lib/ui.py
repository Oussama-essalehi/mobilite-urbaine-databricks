"""Charte graphique (thème sombre) et composants partagés : en-têtes, chiffres clés, graphiques, tableaux."""
from __future__ import annotations

import html
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Palette sombre (ordre fixe des couleurs, validé pour le daltonisme) ---------------------------------
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
BLUE, ORANGE, AQUA = SERIES[0], SERIES[1], SERIES[2]
RED = "#e66767"
MUTED = "#898781"
INK_2 = "#c3c2b7"
GRID = "#2c2c2a"
SURFACE = "#1a1a19"
SEQ_BLUE = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]

# Statuts : manque de vélos côté rouge, manque de places côté bleu, normal en vert
AVAILABILITY = {
    "Vide": "#d03b3b",
    "Presque vide": "#ec835a",
    "Disponible": "#0ca30c",
    "Presque pleine": "#6da7ec",
    "Pleine": "#3987e5",
    "Fermée": "#6b6a65",
}
ACTION = {"REAPPROVISIONNER": "#d03b3b", "DELESTER": "#3987e5", "OK": "#55544f"}
ACTION_LABEL = {"REAPPROVISIONNER": "Ajouter des vélos", "DELESTER": "Retirer des vélos", "OK": "Rien à faire"}
DAY_TYPE = {"Semaine": BLUE, "Week-end": ORANGE}

# Barre d'outils Plotly masquée : ses infobulles sont en anglais
PLOTLY_CONFIG = {"displayModeBar": False, "scrollZoom": False}

CSS = """
<style>
:root { --accent:#3987e5; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#9a988f; --hair:rgba(255,255,255,.10);
        --surface:#1a1a19; }
.block-container { padding-top: 4.6rem; padding-bottom: 3rem; max-width: 1320px; }
h1, h2, h3 { letter-spacing: -0.01em; }
/* éléments d'interface Streamlit en anglais */
[data-testid="stMainMenu"], [data-testid="stStatusWidget"], [data-testid="stToolbarActions"],
[data-testid="stElementToolbar"], [data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="stMetric"] { padding: 14px 16px 12px; border-radius: 12px; background: var(--surface);
                           border: 1px solid var(--hair); }
[data-testid="stMetricLabel"] p { font-size: .85rem; color: var(--ink-2); font-weight: 500; }
[data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 650; }
.page-eyebrow { font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
                color: var(--accent); margin-bottom: .1rem; }
.page-title { font-size: 2.05rem; font-weight: 700; line-height: 1.15; margin: 0 0 .35rem; }
.page-sub { color: var(--ink-2); font-size: 1.05rem; max-width: 880px; margin-bottom: 1.1rem; line-height: 1.55; }
.section { font-size: 1.15rem; font-weight: 650; margin: 1.5rem 0 .15rem; }
.section-sub { color: var(--muted); font-size: .9rem; margin-bottom: .55rem; }
.insight { border-left: 3px solid var(--accent); padding: .65rem 1rem; margin: .5rem 0 1rem;
           background: rgba(57,135,229,.10); border-radius: 0 8px 8px 0; font-size: .97rem; line-height: 1.55; }
.insight .k { font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
              color: var(--accent); display: block; margin-bottom: 2px; }
.hero { padding: 30px 32px 26px; border-radius: 16px; margin-bottom: 1.2rem; border: 1px solid var(--hair);
        background: linear-gradient(135deg, rgba(57,135,229,.20), rgba(25,158,112,.10) 60%, rgba(217,89,38,.08)); }
.hero h1 { font-size: 2.3rem; margin: 0 0 .5rem; line-height: 1.12; }
.hero p { font-size: 1.08rem; color: var(--ink-2); max-width: 880px; margin: 0; line-height: 1.6; }
.steps { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: .4rem 0 1rem; }
@media (max-width: 1000px) { .steps { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.step { border: 1px solid var(--hair); border-radius: 12px; padding: 14px 16px; background: var(--surface); }
.step .n { display:inline-flex; width: 26px; height: 26px; border-radius: 50%; align-items:center; justify-content:center;
           font-weight: 700; font-size: .85rem; color: #111110; margin-bottom: 8px; }
.step .t { font-weight: 650; font-size: 1rem; margin-bottom: 4px; }
.step .d { font-size: .88rem; color: var(--ink-2); line-height: 1.5; }
.card { border: 1px solid var(--hair); border-radius: 12px; padding: 16px 18px; height: 100%; background: var(--surface);
        box-sizing: border-box; }
[data-testid="stColumn"] [data-testid="stMarkdownContainer"]:has(> div > .card) { height: 100%; }
.card .k { font-size: .75rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
.card .v { font-size: 1.6rem; font-weight: 700; margin: 2px 0 4px; }
.card .d { font-size: .9rem; color: var(--ink-2); line-height: 1.5; }
.legend-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
.src-badge { font-size: .8rem; padding: 9px 11px; border-radius: 10px; border: 1px solid var(--hair); line-height: 1.45;
             background: var(--surface); color: var(--ink-2); }
.src-badge b { display:block; font-size: .86rem; color: var(--ink); }
.tbl-wrap { border: 1px solid var(--hair); border-radius: 10px; overflow: auto; background: var(--surface); }
table.tbl { width: 100%; border-collapse: collapse; font-size: .88rem; }
table.tbl th { position: sticky; top: 0; background: #232321; color: var(--ink-2); font-weight: 600; text-align: left;
               padding: 8px 10px; border-bottom: 1px solid var(--hair); white-space: nowrap; }
table.tbl td { padding: 7px 10px; border-bottom: 1px solid rgba(255,255,255,.05); color: var(--ink); }
table.tbl td.num, table.tbl th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.bar { display:flex; align-items:center; gap:8px; min-width: 110px; }
.bar .track { flex: 1; height: 6px; border-radius: 3px; background: rgba(255,255,255,.08); }
.bar .fill { height: 6px; border-radius: 3px; }
.bar .lab { font-variant-numeric: tabular-nums; color: var(--ink-2); font-size: .82rem; width: 38px; text-align: right; }
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


def insight(text: str, label: str = "À retenir") -> None:
    st.markdown(f'<div class="insight"><span class="k">{label}</span>{text}</div>', unsafe_allow_html=True)


def kpis(items: list[tuple]) -> None:
    """items : (libellé, valeur, sous-texte | None, aide | None)"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, delta, help_ = (list(item) + [None] * 4)[:4]
        col.metric(label, value, delta, help=help_, delta_color="off", delta_arrow="off")


def missing(what: str) -> None:
    st.info(f"{what} : ces données ne sont pas encore disponibles. Elles apparaîtront après les premières "
            "exécutions automatiques du traitement.", icon=":material/hourglass_empty:")


def table(df: pd.DataFrame, columns: dict, bars: dict | None = None, height: int | None = None,
          num: tuple = ()) -> None:
    """Tableau HTML en français, sans menus en anglais.

    columns : {colonne: titre} ; bars : {colonne: couleur} pour une jauge 0-1 ; num : colonnes alignées à droite.
    """
    bars = bars or {}
    head = "".join(f'<th class="{"num" if c in num else ""}">{html.escape(t)}</th>' for c, t in columns.items())
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in columns:
            v = r[c]
            if c in bars:
                pct = 0 if pd.isna(v) else max(0.0, min(1.0, float(v)))
                cells.append(f'<td><div class="bar"><div class="track"><div class="fill" style="width:{pct * 100:.0f}%;'
                             f'background:{bars[c]}"></div></div><span class="lab">{pct * 100:.0f} %</span></div></td>')
            else:
                cells.append(f'<td class="{"num" if c in num else ""}">{html.escape("" if pd.isna(v) else str(v))}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    style_ = f' style="max-height:{height}px"' if height else ""
    st.markdown(f'<div class="tbl-wrap"{style_}><table class="tbl"><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>', unsafe_allow_html=True)


# --- Plotly --------------------------------------------------------------------------------------------

def style(fig: go.Figure, height: int = 360, legend: bool = True, hover: str = "x unified") -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36 if legend else 12, b=8),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=13, color=INK_2),
        hovermode=hover, showlegend=legend, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text="",
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=12, bgcolor="#232321", bordercolor=GRID, font_color="#f0efec"),
        separators=", ",
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="", automargin=True, tickfont=dict(color=INK_2),
                     exponentformat="none", separatethousands=True,
                     title_font=dict(size=12, color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True, tickfont=dict(color=INK_2),
                     exponentformat="none", separatethousands=True,
                     title_font=dict(size=12, color=MUTED))
    return fig


def plot(fig: go.Figure, key: str | None = None) -> None:
    # zoom à la molette / au pincement réservé aux cartes, pour ne pas gêner le défilement sur les graphiques
    is_map = any(t.type == "scattermap" for t in fig.data)
    st.plotly_chart(fig, width="stretch", config={**PLOTLY_CONFIG, "scrollZoom": is_map}, key=key)


TOULOUSE = {"lat": 43.6045, "lon": 1.4440}
# Fond de carte : plan Voyager (rues, parcs, Garonne) adouci par un voile ardoise posé sous les stations,
# ni noir ni blanc éblouissant
MAP_STYLE = "carto-voyager"
MAP_TINT = dict(sourcetype="geojson", type="fill", color="#2b3a55", opacity=0.3, below="traces",
                source={"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [
                    [[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]]]}})
MAP_LINE = "rgba(29,36,51,0.75)"
CLUSTER_SLOTS = {"Résidentielle": 0, "Pôle d'activité": 1, "Équilibrée": 2, "Souvent vide": 7, "Souvent pleine": 6}


def cluster_colors(labels) -> dict:
    """Couleur fixe par type de famille (jamais par rang)."""
    out, used = {}, set()
    for lab in sorted({str(x) for x in labels if pd.notna(x)}):
        slot = next((s for k, s in CLUSTER_SLOTS.items() if lab.startswith(k)), None)
        if slot is None or slot in used:
            slot = next(i for i in range(len(SERIES)) if i not in used and i not in CLUSTER_SLOTS.values())
        used.add(slot)
        out[lab] = SERIES[slot]
    return out


def station_map(df: pd.DataFrame, color: str, color_map: dict, hover: dict, size: str | None = None,
                height: int = 540, zoom: float = 11.4, size_max: int = 13, labels: dict | None = None) -> go.Figure:
    import plotly.express as px

    fig = px.scatter_map(
        df, lat="lat", lon="lon", color=color, color_discrete_map=color_map,
        category_orders={color: [c for c in color_map if c in set(df[color])]},
        size=size, size_max=size_max, hover_name="station_label", hover_data=hover, labels=labels or {},
        zoom=zoom, center=TOULOUSE, map_style=MAP_STYLE, height=height,
    )
    if size is None:
        fig.update_traces(marker=dict(size=9))
    fig.update_traces(marker=dict(opacity=0.92))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), font=dict(color=INK_2), map_layers=[MAP_TINT],
                      hoverlabel=dict(bgcolor="#232321", bordercolor=GRID, font_color="#f0efec"),
                      legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.01, title_text="",
                                  bgcolor="rgba(26,26,25,0.88)", font=dict(color="#f0efec")))
    return fig


# --- Formats français -----------------------------------------------------------------------------------

def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt_int(x) -> str:
    return "–" if _bad(x) else f"{x:,.0f}".replace(",", " ")


def fmt_pct(x, digits: int = 0, signed: bool = False) -> str:
    if _bad(x):
        return "–"
    if round(x * 100, digits) == 0:
        x, signed = 0.0, False
    s = f"{x * 100:+.{digits}f}" if signed else f"{x * 100:.{digits}f}"
    return s.replace(".", ",").replace("-", "−") + " %"


def fmt_num(x, digits: int = 2) -> str:
    return "–" if _bad(x) else f"{x:.{digits}f}".replace(".", ",")
