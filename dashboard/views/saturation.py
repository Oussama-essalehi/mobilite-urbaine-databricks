"""Stations saturées aux heures de pointe : classement, carte et profil horaire d'une station."""
import plotly.graph_objects as go
import streamlit as st

from lib import data, ui

ISSUE_COLORS = {"Manque de vélos": "#d03b3b", "Manque de bornes": "#184f95"}


def render() -> None:
    ui.page_header("Analytique · service", "Stations saturées",
                   "Part du temps où chaque station est vide (impossible de louer) ou pleine (impossible de rendre) "
                   "aux heures de pointe de semaine (7-10 h, 17-20 h), sur les 28 derniers jours "
                   "(<code>gold_saturation_ranking</code>).")
    ranking, profile = data.load("saturation_ranking"), data.load("saturation_profile")
    if ranking is None or ranking.empty:
        ui.missing("Classement de saturation")
        return
    ranking["cluster_label"] = ranking.cluster_label.fillna("Non segmentée")

    c1, c2 = st.columns([1, 1.4])
    issue = c1.segmented_control("Problème principal", ["Tous", *ISSUE_COLORS], default="Tous", key="sat_issue") or "Tous"
    ALL = "Tous les segments"
    segment = c2.selectbox("Segment", [ALL, *sorted(ranking.cluster_label.unique())], key="sat_cluster")
    view = ranking if segment == ALL else ranking[ranking.cluster_label == segment]
    if issue != "Tous":
        view = view[view.main_issue == issue]

    ui.kpis([
        ("Souvent vides en pointe", ui.fmt_int((ranking.pct_empty_peak >= .25).sum()), None, "Vides ≥ 25 % du temps"),
        ("Souvent pleines en pointe", ui.fmt_int((ranking.pct_full_peak >= .25).sum()), None, "Pleines ≥ 25 % du temps"),
        ("Saturation moyenne", ui.fmt_pct(ranking.pct_saturated_peak.mean(), 1), None,
         "Moyenne du temps vide + plein aux heures de pointe"),
        ("Station la plus critique", ui.fmt_pct(ranking.pct_saturated_peak.max()), None,
         ranking.loc[ranking.pct_saturated_peak.idxmax(), "station_label"]),
    ])

    left, right = st.columns([1.5, 1])
    with left:
        mp = view.assign(saturation=(view.pct_saturated_peak * 100).round(0))
        fig = ui.station_map(mp, "main_issue", ISSUE_COLORS, size="pct_saturated_peak", height=520, size_max=16,
                             hover={"saturation": ":.0f", "cluster_label": True, "pct_saturated_peak": False,
                                    "main_issue": False, "lat": False, "lon": False})
        ui.plot(fig, key="sat_map")
        st.caption("Taille du point = part du temps saturé en pointe (%).")
    with right:
        ui.section("Top 15", "Classement par temps saturé en pointe")
        st.dataframe(
            view.nsmallest(15, "saturation_rank")[["saturation_rank", "station_label", "pct_saturated_peak"]],
            hide_index=True, height=520, width="stretch", column_config={
                "saturation_rank": st.column_config.NumberColumn("#", width=40),
                "station_label": "Station",
                "pct_saturated_peak": st.column_config.ProgressColumn("Saturé", format="percent", min_value=0,
                                                                      max_value=1)})

    if profile is None or profile.empty:
        return
    ui.section("Profil horaire d'une station", "Moyenne sur 28 jours, heures où la station est ouverte")
    labels = ranking.sort_values("saturation_rank").set_index("station_id").station_label
    sid = st.selectbox("Station", labels.index, format_func=lambda i: labels[i], key="sat_station")
    p = profile[profile.station_id == sid]
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for dt in ["Semaine", "Week-end"]:
            d = p[p.day_type == dt].sort_values("hour_of_day")
            fig.add_scatter(x=d.hour_of_day, y=d.avg_fill_rate, name=dt, mode="lines",
                            line=dict(color=ui.DAY_TYPE[dt], width=2.5), hovertemplate="%{y:.0%}<extra>" + dt + "</extra>")
        for h0, h1 in [(7, 10), (17, 20)]:
            fig.add_vrect(x0=h0, x1=h1, fillcolor=ui.MUTED, opacity=.08, line_width=0)
        fig.update_yaxes(title_text="Taux de remplissage", tickformat=".0%", range=[0, 1])
        fig.update_xaxes(title_text="Heure (zones grisées : pointes de semaine)", dtick=3, ticksuffix=" h")
        ui.plot(ui.style(fig, height=330), key="sat_fill")
    with right:
        d = p[p.day_type == "Semaine"].sort_values("hour_of_day")
        fig = go.Figure()
        fig.add_scatter(x=d.hour_of_day, y=d.pct_time_empty, name="Vide", mode="lines",
                        line=dict(color=ISSUE_COLORS["Manque de vélos"], width=2.5), hovertemplate="%{y:.0%}<extra>Vide</extra>")
        fig.add_scatter(x=d.hour_of_day, y=d.pct_time_full, name="Pleine", mode="lines",
                        line=dict(color=ISSUE_COLORS["Manque de bornes"], width=2.5), hovertemplate="%{y:.0%}<extra>Pleine</extra>")
        fig.update_yaxes(title_text="Part du temps (semaine)", tickformat=".0%", rangemode="tozero")
        fig.update_xaxes(title_text="Heure", dtick=3, ticksuffix=" h")
        ui.plot(ui.style(fig, height=330), key="sat_pct")

    r = ranking.set_index("station_id").loc[sid]
    ui.insight(f"<b>{r.station_label}</b> ({r.cluster_label}) : vide {ui.fmt_pct(r.pct_empty_peak)} et pleine "
               f"{ui.fmt_pct(r.pct_full_peak)} du temps aux pointes de semaine — rang {int(r.saturation_rank)} "
               f"sur {len(ranking)}. Problème principal : <b>{r.main_issue.lower()}</b>.")
    ui.data_table(view.sort_values("saturation_rank")[[
        "saturation_rank", "station_label", "cluster_label", "main_issue", "pct_empty_peak", "pct_full_peak",
        "avg_net_flow_morning", "avg_net_flow_evening"]])
