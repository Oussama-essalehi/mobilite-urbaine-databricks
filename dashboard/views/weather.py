"""Effet de la météo : départs relatifs par condition et par température, à heure comparable."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib import data, ui

WEATHER_ORDER = ["Dégagé", "Nuageux", "Pluie", "Forte pluie"]
DRYNESS = {"Dégagé": "Temps sec", "Nuageux": "Temps sec", "Pluie": "Pluie", "Forte pluie": "Forte pluie"}
DRY_COLORS = {"Temps sec": ui.BLUE, "Pluie": ui.ORANGE, "Forte pluie": ui.AQUA}


def effect_bars(df, key):
    rel = df.relative_to_reference - 1
    fig = go.Figure(go.Bar(
        x=df.category.str.replace(r"^\d\. ", "", regex=True), y=rel,
        marker=dict(color=np.where(rel < -0.005, "#d03b3b", np.where(rel > 0.005, ui.BLUE, ui.MUTED)), cornerradius=4),
        text=[ui.fmt_pct(v, signed=True) for v in rel], textposition="outside", cliponaxis=False,
        customdata=np.c_[df.n_hours, df.avg_departures_per_hour],
        hovertemplate="%{x} : %{y:+.1%} vs référence<br>%{customdata[0]} heures · "
                      "%{customdata[1]:,.0f} départs/h<extra></extra>"))
    lo, hi = min(rel.min(), 0), max(rel.max(), 0)
    pad = (hi - lo) * .15 or .05
    fig.update_yaxes(tickformat="+.0%", zeroline=True, zerolinecolor=ui.MUTED, title_text="Écart de départs",
                     range=[lo - pad, hi + pad])
    fig.update_layout(bargap=.45)
    ui.plot(ui.style(fig, height=330, legend=False, hover="closest"), key=key)


def render() -> None:
    ui.page_header("Analytique · contexte", "Effet de la météo",
                   "Pour neutraliser l'effet horaire, chaque heure est comparée à la moyenne des heures <b>sèches</b> "
                   "de même type de jour et de même heure (table <code>gold_weather_effect</code>). "
                   "Une valeur de −30 % signifie 30 % de départs en moins qu'une heure sèche comparable.")
    effect, net = data.load("weather_effect"), data.load("network_hourly")
    if effect is None or effect.empty:
        ui.missing("Effet météo")
        return

    meteo = effect[effect.dimension == "Météo"].set_index("category").reindex(WEATHER_ORDER).dropna(subset=["n_hours"]).reset_index()
    temp = effect[effect.dimension == "Température"].sort_values("category")
    get = meteo.set_index("category").relative_to_reference
    best_band = temp.loc[temp.relative_to_reference.idxmax()]
    ui.kpis([
        ("Effet de la pluie", ui.fmt_pct(get.get("Pluie", np.nan) - 1, signed=True), None, "Précipitations > 0,1 mm/h"),
        ("Effet de la forte pluie", ui.fmt_pct(get.get("Forte pluie", np.nan) - 1, signed=True), None, "≥ 2 mm/h"),
        ("Tranche la plus favorable", best_band.category[3:], None, "Température associée au plus de départs relatifs"),
        ("Heures analysées", ui.fmt_int(meteo.n_hours.sum()), None, "Heures réseau avec une observation météo"),
    ])

    left, right = st.columns(2)
    with left:
        ui.section("Par condition météo")
        effect_bars(meteo, "w_cat")
    with right:
        ui.section("Par tranche de température")
        effect_bars(temp, "w_temp")

    if net is None or net.empty:
        return
    base = net[net.weather_category.notna() & (net.stations_reporting > 0)].copy()
    ref = base[base.weather_category.isin(["Dégagé", "Nuageux"])].groupby(["day_type", "hour_of_day"]).total_departures.mean()
    base = base.join(ref.rename("ref"), on=["day_type", "hour_of_day"])
    base = base[(base.ref > 50)]                       # heures creuses de nuit : ratio trop bruité
    base["ratio"] = base.total_departures / base.ref - 1
    base["meteo"] = base.weather_category.map(DRYNESS)

    ui.section("Chaque heure face à sa référence", "Une heure = un point ; heures de nuit à très faible trafic exclues")
    fig = go.Figure()
    for k, color in DRY_COLORS.items():
        d = base[base.meteo == k]
        fig.add_scatter(x=d.temperature, y=d.ratio, mode="markers", name=f"{k} ({len(d)} h)",
                        marker=dict(color=color, size=8 if k != "Temps sec" else 6,
                                    opacity=.85 if k != "Temps sec" else .35, line=dict(width=1, color="white")),
                        customdata=np.c_[d.hour_local.dt.strftime("%d/%m %Hh"), d.total_departures],
                        hovertemplate="%{customdata[0]} · %{x:.1f} °C<br>%{y:+.0%} vs référence · "
                                      "%{customdata[1]:,.0f} départs<extra>" + k + "</extra>")
    fig.update_xaxes(title_text="Température (°C)", ticksuffix=" °C")
    fig.update_yaxes(title_text="Écart à la référence sèche", tickformat="+.0%", zeroline=True, zerolinecolor=ui.MUTED)
    ui.plot(ui.style(fig, height=420, hover="closest"), key="w_scatter")

    rain_gap = get.get("Pluie", np.nan) - 1
    ui.insight(f"La pluie fait chuter la demande de <b>{ui.fmt_pct(-rain_gap)}</b> à heure comparable : "
               "c'est pourquoi la prévision Open-Meteo est une variable du modèle de prévision, et la raison pour "
               "laquelle les besoins de rééquilibrage baissent les jours de pluie.")
    ui.data_table(effect)
