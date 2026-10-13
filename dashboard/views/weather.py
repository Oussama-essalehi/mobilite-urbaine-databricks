"""L'effet de la météo : trajets en plus ou en moins selon la pluie et la température."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib import data, ui

WEATHER_ORDER = ["Dégagé", "Nuageux", "Pluie", "Forte pluie"]
WEATHER_LABELS = {"Dégagé": "Ciel dégagé", "Nuageux": "Nuageux", "Pluie": "Pluie", "Forte pluie": "Forte pluie"}


def effect_bars(df, labels, key):
    rel = df.relative_to_reference - 1
    fig = go.Figure(go.Bar(
        x=labels, y=rel,
        marker=dict(color=np.where(rel < -0.005, "#d03b3b", np.where(rel > 0.005, ui.BLUE, ui.MUTED)), cornerradius=4),
        text=[ui.fmt_pct(v, signed=True) for v in rel], textposition="outside", cliponaxis=False,
        hovertemplate="%{x} : %{text} de trajets<extra></extra>"))
    lo, hi = min(rel.min(), 0), max(rel.max(), 0)
    pad = (hi - lo) * .18 or .05
    fig.update_yaxes(tickformat="+.0%", zeroline=True, zerolinecolor=ui.MUTED, range=[lo - pad, hi + pad],
                     title_text="Trajets en plus ou en moins")
    fig.update_layout(bargap=.45)
    ui.plot(ui.style(fig, height=340, legend=False, hover="closest"), key=key)


def render() -> None:
    ui.page_header("Comprendre", "L'effet de la météo",
                   "Pour mesurer l'effet de la météo de façon juste, chaque heure est comparée à une heure "
                   "<b>sans pluie</b> du même type (même heure, semaine ou week-end). Exemple : −30 % signifie "
                   "30 % de trajets en moins qu'à la même heure par temps sec.")
    effect = data.load("weather_effect")
    if effect is None or effect.empty:
        ui.missing("L'effet de la météo")
        return

    meteo = (effect[effect.dimension == "Météo"].set_index("category").reindex(WEATHER_ORDER)
             .dropna(subset=["n_hours"]).reset_index())
    temp = effect[effect.dimension == "Température"].sort_values("category")
    rel = meteo.set_index("category").relative_to_reference
    best = temp.loc[temp.relative_to_reference.idxmax()]
    ui.kpis([
        ("Quand il pleut", ui.fmt_pct(rel.get("Pluie", np.nan) - 1, signed=True), "de trajets", None),
        ("Quand il pleut fort", ui.fmt_pct(rel.get("Forte pluie", np.nan) - 1, signed=True), "de trajets",
         "Plus de 2 mm de pluie en une heure"),
        ("Température idéale", best.category[3:], "le plus de trajets", None),
    ])

    left, right = st.columns(2)
    with left:
        ui.section("Selon le temps qu'il fait")
        effect_bars(meteo, meteo.category.map(WEATHER_LABELS), "w_cat")
    with right:
        ui.section("Selon la température")
        effect_bars(temp, temp.category.str[3:], "w_temp")

    ui.insight(f"La pluie fait baisser les trajets d'environ <b>{ui.fmt_pct(1 - rel.get('Pluie', np.nan))}</b>. "
               "C'est pour cela que la météo prévue fait partie des informations utilisées pour prévoir l'affluence, "
               "et qu'il y a moins de vélos à déplacer les jours de pluie.")
