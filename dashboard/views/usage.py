"""Quand roule-t-on ? Trajets par jour, journée type, jours × heures, stations les plus utilisées."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib import data, ui

PERIODS = {"7 derniers jours": 7, "14 derniers jours": 14, "28 derniers jours": 28}
WEEKDAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]


def render() -> None:
    ui.page_header("Comprendre", "Quand roule-t-on ?",
                   "Le nombre de trajets est estimé en comptant les vélos qui quittent chaque station. "
                   "Le chiffre réel est un peu plus élevé : un vélo pris puis remplacé entre deux mesures ne se voit pas.")

    net, daily_st, dim = data.load("network_hourly"), data.load("station_daily"), data.load("dim_station")
    if net is None or net.empty:
        ui.missing("Les trajets")
        return

    period = st.segmented_control("Période", list(PERIODS), default="28 derniers jours", key="usage_period") \
        or "28 derniers jours"
    today = net.date_local.max()                       # journée en cours, incomplète : exclue
    full = net[net.date_local < today]
    start = today - pd.Timedelta(days=PERIODS[period])
    cur = full[full.date_local >= start]
    span = max(cur.date_local.nunique(), 1)
    by_day = cur.groupby(["date_local", "day_type"]).total_departures.sum().reset_index()
    wd = cur[cur.day_type == "Semaine"].groupby("hour_of_day").total_departures.mean()

    ui.kpis([
        ("Trajets sur la période", ui.fmt_int(cur.total_departures.sum()), f"sur {span} jours", None),
        ("Un jour de semaine", ui.fmt_int(by_day[by_day.day_type == "Semaine"].total_departures.mean()),
         "trajets en moyenne", None),
        ("Un jour de week-end", ui.fmt_int(by_day[by_day.day_type == "Week-end"].total_departures.mean()),
         "trajets en moyenne", None),
        ("Heure la plus chargée", f"{wd.idxmax()} h" if not wd.empty else "–", "en semaine", None),
    ])

    ui.section("Trajets jour par jour", "En dessous : la pluie tombée ce jour-là, en millimètres.")
    rain = cur.groupby("date_local").precipitation.sum()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.76, .24], vertical_spacing=.06)
    for dt in ["Semaine", "Week-end"]:
        d = by_day[by_day.day_type == dt]
        fig.add_bar(x=d.date_local, y=d.total_departures, name=dt, marker=dict(color=ui.DAY_TYPE[dt], cornerradius=3),
                    hovertemplate="%{x|%d/%m} : %{y:,.0f} trajets<extra>" + dt + "</extra>", row=1, col=1)
    fig.add_bar(x=rain.index, y=rain.values, name="Pluie", marker=dict(color=ui.MUTED, cornerradius=2),
                hovertemplate="%{x|%d/%m} : %{y:.1f} mm de pluie<extra></extra>", showlegend=False, row=2, col=1)
    fig.update_yaxes(title_text="Trajets", row=1, col=1)
    fig.update_yaxes(title_text="Pluie (mm)", row=2, col=1)
    fig.update_xaxes(tickformat="%d/%m")
    fig.update_layout(bargap=.18)
    ui.plot(ui.style(fig, height=380, hover="closest"), key="usage_daily")

    left, right = st.columns(2)
    with left:
        ui.section("Une journée type", "Nombre moyen de trajets selon l'heure")
        prof = cur.groupby(["day_type", "hour_of_day"]).total_departures.mean().reset_index()
        fig = go.Figure()
        for dt in ["Semaine", "Week-end"]:
            d = prof[prof.day_type == dt]
            fig.add_scatter(x=d.hour_of_day, y=d.total_departures, name=dt, mode="lines",
                            line=dict(color=ui.DAY_TYPE[dt], width=2.5),
                            hovertemplate="%{y:,.0f} trajets<extra>" + dt + "</extra>")
        fig.update_xaxes(title_text="Heure de la journée", dtick=3, ticksuffix=" h")
        fig.update_yaxes(title_text="Trajets par heure", rangemode="tozero")
        ui.plot(ui.style(fig, height=340), key="usage_profile")
    with right:
        ui.section("Quels jours, à quelles heures ?", "Plus la case est claire, plus il y a de trajets")
        heat = cur.groupby(["weekday", "hour_of_day"]).total_departures.mean().unstack().reindex(range(1, 8))
        fig = go.Figure(go.Heatmap(
            z=heat.values, x=heat.columns, y=WEEKDAYS, colorscale=ui.SEQ_BLUE, xgap=2, ygap=2,
            colorbar=dict(title=dict(text="Trajets / h", side="right"), thickness=10),
            hovertemplate="%{y} à %{x} h : %{z:,.0f} trajets<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(dtick=3, ticksuffix=" h")
        ui.plot(ui.style(fig, height=340, legend=False, hover="closest"), key="usage_heat")

    if not wd.empty:
        evening = wd.loc[16:20]
        ui.insight(f"En semaine, deux pics : le matin vers <b>{wd.loc[:12].idxmax()} h</b> et le soir vers "
                   f"<b>{evening.idxmax() if not evening.empty else '18'} h</b>, aux heures des trajets domicile-travail. "
                   "Le week-end, l'usage est plus étalé et culmine l'après-midi.")

    if daily_st is None or dim is None:
        return
    st_cur = daily_st[(daily_st.date_local >= start) & (daily_st.date_local < today)]
    agg = (st_cur.groupby("station_id").departures.sum().div(span).rename("per_day").reset_index()
           .merge(dim[["station_id", "station_label"]], on="station_id"))
    ui.section("Les 10 stations les plus utilisées", "Nombre moyen de vélos pris par jour")
    top = agg.nlargest(10, "per_day").sort_values("per_day")
    fig = go.Figure(go.Bar(y=top.station_label.str.replace(r"^\d+ - ", "", regex=True), x=top.per_day,
                           orientation="h", marker=dict(color=ui.BLUE, cornerradius=4),
                           text=top.per_day.round(0).astype(int), textposition="outside", cliponaxis=False,
                           hovertemplate="%{y} : %{x:.0f} vélos pris par jour<extra></extra>"))
    fig.update_xaxes(showticklabels=False, showgrid=False, range=[0, top.per_day.max() * 1.12])
    ui.plot(ui.style(fig, height=380, legend=False, hover="closest"), key="usage_top")
