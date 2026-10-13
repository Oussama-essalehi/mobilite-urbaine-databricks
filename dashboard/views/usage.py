"""Fréquentation : départs estimés, profils horaires, carte de chaleur, stations les plus actives."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib import data, ui

PERIODS = {"7 jours": 7, "14 jours": 14, "28 jours": 28, "Tout l'historique": None}
WEEKDAYS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]


def render() -> None:
    ui.page_header("Analytique · demande", "Fréquentation du réseau",
                   "Départs et arrivées estimés à partir des variations de stock entre deux collectes (≤ 30 min). "
                   "C'est un minorant : un départ et une arrivée dans le même intervalle s'annulent.")

    net, daily_st, dim = data.load("network_hourly"), data.load("station_daily"), data.load("dim_station")
    if net is None or net.empty:
        ui.missing("Vue réseau horaire")
        return

    period = st.segmented_control("Période", list(PERIODS), default="28 jours", key="usage_period") or "28 jours"
    today = net.date_local.max()                       # journée en cours, incomplète : exclue des totaux
    full = net[net.date_local < today]
    days = PERIODS[period]
    start = full.date_local.min() if days is None else today - pd.Timedelta(days=days)
    cur = full[full.date_local >= start]
    span = cur.date_local.nunique()
    prev = full[(full.date_local >= start - pd.Timedelta(days=span)) & (full.date_local < start)]

    dep = cur.total_departures.sum()
    delta = None
    if prev.date_local.nunique() == span and prev.total_departures.sum() > 0:
        delta = ui.fmt_pct(dep / prev.total_departures.sum() - 1, 1, signed=True) + " vs période précédente"
    wd = cur[cur.day_type == "Semaine"].groupby("hour_of_day").total_departures.mean()
    by_day = cur.groupby(["date_local", "day_type"]).total_departures.sum().reset_index()
    ui.kpis([
        ("Départs estimés", ui.fmt_int(dep), delta, f"{span} jours complets"),
        ("Moyenne par jour", ui.fmt_int(dep / max(span, 1)), None, None),
        ("Semaine / jour", ui.fmt_int(by_day[by_day.day_type == "Semaine"].total_departures.mean()), None, None),
        ("Week-end / jour", ui.fmt_int(by_day[by_day.day_type == "Week-end"].total_departures.mean()), None, None),
        ("Pic en semaine", f"{wd.idxmax()} h" if not wd.empty else "–", None,
         "Heure où la moyenne des départs est la plus haute"),
    ])

    # --- évolution journalière + pluie (deux panneaux, même axe temporel)
    ui.section("Départs par jour", "Les jours de pluie se lisent directement sur le panneau du bas.")
    rain = cur.groupby("date_local").precipitation.sum()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.76, .24], vertical_spacing=.06)
    for dt in ["Semaine", "Week-end"]:
        d = by_day[by_day.day_type == dt]
        fig.add_bar(x=d.date_local, y=d.total_departures, name=dt, marker=dict(color=ui.DAY_TYPE[dt], cornerradius=3),
                    hovertemplate="%{x|%d/%m} : %{y:,.0f} départs<extra>" + dt + "</extra>", row=1, col=1)
    fig.add_bar(x=rain.index, y=rain.values, name="Pluie (mm)", marker=dict(color=ui.MUTED, cornerradius=2),
                hovertemplate="%{x|%d/%m} : %{y:.1f} mm<extra></extra>", showlegend=False, row=2, col=1)
    fig.update_yaxes(title_text="Départs", row=1, col=1)
    fig.update_yaxes(title_text="Pluie (mm)", row=2, col=1)
    fig.update_layout(bargap=.18)
    fig.update_xaxes(tickformat="%d/%m")
    ui.plot(ui.style(fig, height=400, hover="closest"), key="usage_daily")

    left, right = st.columns(2)
    with left:
        ui.section("Profil horaire moyen", "Départs par heure : double pointe domicile-travail en semaine, après-midi le week-end")
        prof = cur.groupby(["day_type", "hour_of_day"]).total_departures.mean().reset_index()
        fig = go.Figure()
        for dt in ["Semaine", "Week-end"]:
            d = prof[prof.day_type == dt]
            fig.add_scatter(x=d.hour_of_day, y=d.total_departures, name=dt, mode="lines",
                            line=dict(color=ui.DAY_TYPE[dt], width=2.5),
                            hovertemplate="%{y:,.0f} départs/h<extra>" + dt + "</extra>")
        fig.update_xaxes(title_text="Heure", dtick=3, ticksuffix=" h")
        fig.update_yaxes(title_text="Départs / heure", rangemode="tozero")
        ui.plot(ui.style(fig, height=340), key="usage_profile")
    with right:
        ui.section("Carte de chaleur jour × heure", "Départs moyens par heure")
        heat = cur.groupby(["weekday", "hour_of_day"]).total_departures.mean().unstack().reindex(range(1, 8))
        fig = go.Figure(go.Heatmap(
            z=heat.values, x=heat.columns, y=WEEKDAYS, colorscale=ui.SEQ_BLUE, xgap=2, ygap=2,
            colorbar=dict(title=dict(text="Départs/h", side="right"), thickness=10),
            hovertemplate="%{y} %{x} h : %{z:,.0f} départs<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(dtick=3, ticksuffix=" h")
        ui.plot(ui.style(fig, height=340, legend=False, hover="closest"), key="usage_heat")

    if daily_st is None or dim is None:
        return
    st_cur = daily_st[(daily_st.date_local >= start) & (daily_st.date_local < today)]
    agg = (st_cur.groupby("station_id")[["departures", "arrivals"]].sum()
           .join(dim.set_index("station_id")[["station_label", "cluster_label"]]).reset_index())
    agg["net"] = agg.arrivals - agg.departures
    agg["per_day"] = agg.departures / max(span, 1)

    left, right = st.columns(2)
    with left:
        ui.section("Stations les plus actives", "Départs moyens par jour")
        top = agg.nlargest(12, "per_day").sort_values("per_day")
        fig = go.Figure(go.Bar(y=top.station_label, x=top.per_day, orientation="h",
                               marker=dict(color=ui.BLUE, cornerradius=4),
                               text=top.per_day.round(0).astype(int), textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>%{x:.0f} départs/jour<extra></extra>"))
        fig.update_xaxes(showticklabels=False, showgrid=False)
        ui.plot(ui.style(fig, height=420, legend=False, hover="closest"), key="usage_top")
    with right:
        ui.section("Déséquilibre arrivées − départs", "Les stations qui accumulent (bleu) ou perdent (rouge) des vélos")
        ext = pd.concat([agg.nsmallest(6, "net"), agg.nlargest(6, "net")]).sort_values("net")
        ext["per_day"] = ext.net / max(span, 1)
        fig = go.Figure(go.Bar(y=ext.station_label, x=ext.per_day, orientation="h",
                               marker=dict(color=np.where(ext.per_day < 0, "#d03b3b", ui.BLUE), cornerradius=4),
                               hovertemplate="%{y}<br>%{x:+.1f} vélos/jour<extra></extra>"))
        fig.update_xaxes(title_text="Flux net moyen (vélos / jour)", zeroline=True, zerolinecolor=ui.MUTED)
        ui.plot(ui.style(fig, height=420, legend=False, hover="closest"), key="usage_net")
    ui.insight("Ces déséquilibres structurels sont compensés chaque jour par les camions de rééquilibrage : "
               "ils sont inclus dans les flux estimés, et c'est exactement ce que la page <b>Rééquilibrage</b> cherche à anticiper.")
    ui.data_table(agg.sort_values("departures", ascending=False)[
        ["station_label", "cluster_label", "departures", "arrivals", "net"]])
