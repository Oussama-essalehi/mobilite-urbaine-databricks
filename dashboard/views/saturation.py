"""Stations à problème : les stations souvent vides ou pleines aux heures de pointe."""
import plotly.graph_objects as go
import streamlit as st

from lib import data, ui

ISSUE_COLORS = {"Manque de vélos": "#d03b3b", "Manque de places": "#3987e5"}


def render() -> None:
    ui.page_header("Comprendre", "Stations à problème",
                   "Aux heures de pointe en semaine (7 h – 10 h et 17 h – 20 h), certaines stations sont souvent "
                   "<b>vides</b> (impossible de prendre un vélo) ou <b>pleines</b> (impossible d'en rendre un). "
                   "Calcul sur les 4 dernières semaines.")
    ranking, profile = data.load("saturation_ranking"), data.load("saturation_profile")
    if ranking is None or ranking.empty:
        ui.missing("Le classement des stations")
        return
    ranking["issue"] = ranking.main_issue.replace({"Manque de bornes": "Manque de places"})
    ranking["short"] = ranking.station_label.str.replace(r"^\d+ - ", "", regex=True)

    worst = ranking.loc[ranking.pct_saturated_peak.idxmax()]
    ui.kpis([
        ("Souvent vides", ui.fmt_int((ranking.pct_empty_peak >= .25).sum()), "stations, plus d'un quart du temps", None),
        ("Souvent pleines", ui.fmt_int((ranking.pct_full_peak >= .25).sum()), "stations, plus d'un quart du temps", None),
        ("La plus touchée", ui.fmt_pct(worst.pct_saturated_peak), f"du temps · {worst.short}", None),
    ])

    issue = st.segmented_control("Afficher", ["Toutes", *ISSUE_COLORS], default="Toutes", key="sat_issue") or "Toutes"
    view = ranking if issue == "Toutes" else ranking[ranking.issue == issue]

    left, right = st.columns([1.5, 1])
    with left:
        mp = view.assign(part=(view.pct_saturated_peak * 100).round(0))
        fig = ui.station_map(mp, "issue", ISSUE_COLORS, size="pct_saturated_peak", height=520, size_max=16,
                             hover={"part": ":.0f", "pct_saturated_peak": False, "issue": False, "lat": False,
                                    "lon": False}, labels={"part": "Vide ou pleine (% du temps)"})
        ui.plot(fig, key="sat_map")
        st.caption("Plus le point est gros, plus la station est souvent vide ou pleine.")
    with right:
        ui.section("Les 15 stations les plus touchées", "Part du temps où la station est vide ou pleine")
        top = view.nsmallest(15, "saturation_rank")
        ui.table(top, {"short": "Station", "issue": "Problème", "pct_saturated_peak": "Vide ou pleine"},
                 bars={"pct_saturated_peak": ui.ORANGE}, height=480)

    if profile is None or profile.empty:
        return
    ui.section("Zoom sur une station", "Remplissage moyen au fil de la journée : 0 % = vide, 100 % = pleine")
    labels = ranking.sort_values("saturation_rank").set_index("station_id").short
    sid = st.selectbox("Choisir une station", labels.index, format_func=lambda i: labels[i], key="sat_station")
    p = profile[profile.station_id == sid]
    fig = go.Figure()
    for h0, h1 in [(7, 10), (17, 20)]:
        fig.add_vrect(x0=h0, x1=h1, fillcolor="#ffffff", opacity=.05, line_width=0)
    for dt in ["Semaine", "Week-end"]:
        d = p[p.day_type == dt].sort_values("hour_of_day")
        fig.add_scatter(x=d.hour_of_day, y=d.avg_fill_rate, name=dt, mode="lines",
                        line=dict(color=ui.DAY_TYPE[dt], width=2.5), hovertemplate="%{y:.0%}<extra>" + dt + "</extra>")
    fig.update_yaxes(title_text="Remplissage", tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title_text="Heure (zones plus claires : heures de pointe)", dtick=2, ticksuffix=" h")
    ui.plot(ui.style(fig, height=330), key="sat_fill")

    r = ranking.set_index("station_id").loc[sid]
    ui.insight(f"<b>{r.short}</b> est vide {ui.fmt_pct(r.pct_empty_peak)} du temps et pleine "
               f"{ui.fmt_pct(r.pct_full_peak)} du temps aux heures de pointe. Elle se classe "
               f"<b>{int(r.saturation_rank)}<sup>e</sup></b> sur {len(ranking)} stations.")
