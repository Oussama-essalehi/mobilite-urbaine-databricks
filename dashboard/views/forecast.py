"""Prévisions 24 heures : combien de vélos vont partir et arriver, et quelle confiance accorder au modèle."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data, ui

TARGETS = {"Vélos qui partent": "departures", "Vélos qui arrivent": "arrivals"}
FEATURE_LABELS = {
    "station_hour_mean": "Habitudes de la station à cette heure", "station_mean": "Activité habituelle de la station",
    "hour_of_day": "Heure de la journée", "lag_24h": "Même heure la veille", "lag_168h": "Même heure la semaine passée",
    "weekday": "Jour de la semaine", "is_weekend": "Semaine ou week-end", "temperature": "Température",
    "precipitation": "Pluie", "wind_speed": "Vent", "capacity": "Taille de la station", "cluster_id": "Famille de la station",
}


def render() -> None:
    ui.page_header("Prévoir", "Prévisions pour les 24 prochaines heures",
                   "Un modèle de prévision a appris, à partir de l'historique, comment l'affluence varie selon l'heure, "
                   "le jour, la météo et les habitudes de chaque station. Il estime ensuite, pour chaque station et "
                   "chaque heure, combien de vélos vont partir et arriver.")
    metrics, fc, net = data.load("model_metrics"), data.load("forecast_station_hourly"), data.load("network_hourly")
    if metrics is None or metrics.empty:
        ui.missing("Les prévisions (il faut au moins 14 jours de données)")
        return

    label = st.segmented_control("Prévoir", list(TARGETS), default="Vélos qui partent", key="fc_target") \
        or "Vélos qui partent"
    target = TARGETS[label]
    last = metrics[metrics.target == target].sort_values("trained_at").iloc[-1]
    ui.kpis([
        ("Erreur moyenne", f"± {ui.fmt_num(last.mae, 2)}", "vélo par station et par heure",
         "Écart moyen entre la prévision et la réalité, mesuré sur les 7 derniers jours"),
        ("Méthode simple", f"± {ui.fmt_num(last.baseline_mae, 2)}", "même heure, semaine passée",
         "Erreur obtenue en recopiant simplement ce qui s'est passé une semaine plus tôt"),
        ("Gain du modèle", ui.fmt_pct(last.skill_vs_baseline), "d'erreur en moins", None),
        ("Mis à jour le", f"{pd.Timestamp(last.trained_at):%d/%m}", f"à {pd.Timestamp(last.trained_at):%H:%M}",
         "Le modèle est réentraîné automatiquement chaque nuit"),
    ])

    if fc is not None and not fc.empty:
        ui.section("Les 24 prochaines heures, toute la ville",
                   "En bleu la prévision, en gris ce qui s'est passé à la même heure la semaine dernière")
        f = fc.groupby("hour_local")[f"pred_{target}"].sum().reset_index()
        fig = go.Figure()
        if net is not None:
            obs = net.set_index("hour_local")[f"total_{target}"]
            f["j7"] = (f.hour_local - pd.Timedelta(days=7)).map(obs)
            fig.add_scatter(x=f.hour_local, y=f.j7, name="La semaine dernière", mode="lines",
                            line=dict(color=ui.MUTED, width=2), hovertemplate="%{y:,.0f}<extra>Semaine dernière</extra>")
        fig.add_scatter(x=f.hour_local, y=f[f"pred_{target}"], name="Prévision", mode="lines",
                        line=dict(color=ui.BLUE, width=2.5), fill="tozeroy", fillcolor="rgba(57,135,229,0.14)",
                        hovertemplate="%{y:,.0f}<extra>Prévision</extra>")
        fig.update_yaxes(title_text="Vélos par heure", rangemode="tozero")
        fig.update_xaxes(tickformat="%d/%m %Hh", hoverformat="%d/%m à %Hh")
        ui.plot(ui.style(fig, height=340), key="fc_next")

    test = data.load("forecast_test_network")
    ui.section("Le modèle est-il fiable ?", "Prévision et réalité sur les 7 derniers jours, pour toute la ville")
    if test is None:
        st.info("Cette comparaison est conservée dans l'historique d'entraînement du modèle, et n'est pas affichée ici.",
                icon=":material/info:")
    else:
        t = test[test.target == target]
        fig = go.Figure()
        fig.add_scatter(x=t.hour_local, y=t.actual, name="Réalité", mode="lines", line=dict(color=ui.MUTED, width=2),
                        hovertemplate="%{y:,.0f}<extra>Réalité</extra>")
        fig.add_scatter(x=t.hour_local, y=t.pred, name="Prévision", mode="lines", line=dict(color=ui.BLUE, width=2),
                        hovertemplate="%{y:,.0f}<extra>Prévision</extra>")
        fig.update_yaxes(title_text="Vélos par heure")
        fig.update_xaxes(tickformat="%d/%m", hoverformat="%d/%m à %Hh")
        ui.plot(ui.style(fig, height=320), key="fc_test")
        ui.insight("Les deux courbes se superposent presque partout : le modèle reproduit bien les pics du matin et du "
                   f"soir. Son erreur est <b>{ui.fmt_pct(last.skill_vs_baseline)}</b> plus faible que celle d'une "
                   "méthode simple qui recopierait la semaine précédente.")

    if fc is not None and not fc.empty:
        dim = data.load("dim_station")
        ui.section("Prévision pour une station")
        labels = (dim.set_index("station_id").station_label.str.replace(r"^\d+ - ", "", regex=True)
                  if dim is not None else pd.Series(dtype=str))
        busiest = fc.groupby("station_id").pred_departures.sum().sort_values(ascending=False).index
        sid = st.selectbox("Choisir une station", busiest, format_func=lambda i: labels.get(i, f"Station {i}"),
                           key="fc_station")
        s = fc[fc.station_id == sid].sort_values("hour_local")
        fig = go.Figure()
        fig.add_bar(x=s.hour_local, y=s.pred_departures, name="Vélos qui partent", marker=dict(color=ui.BLUE, cornerradius=3))
        fig.add_bar(x=s.hour_local, y=s.pred_arrivals, name="Vélos qui arrivent", marker=dict(color=ui.ORANGE, cornerradius=3))
        fig.update_layout(barmode="group", bargap=.25, bargroupgap=.08)
        fig.update_yaxes(title_text="Vélos par heure")
        fig.update_xaxes(tickformat="%d/%m %Hh")
        ui.plot(ui.style(fig, height=300), key="fc_station_chart")
        net_flow = s.pred_net_flow.sum()
        trend = "devrait plutôt se remplir" if net_flow > 0 else "devrait plutôt se vider"
        ui.insight(f"Sur les 24 prochaines heures, cette station {trend} "
                   f"(<b>{'+' if net_flow > 0 else '−'}{ui.fmt_num(abs(net_flow), 0)} vélos</b> au total).")

    imp = data.load("feature_importance")
    if imp is not None:
        with st.expander("Pour les curieux : sur quoi le modèle s'appuie-t-il ?", icon=":material/lightbulb:"):
            st.markdown("Plus la barre est longue, plus l'information compte pour la prévision : on la mesure en "
                        "brouillant volontairement cette information et en regardant de combien l'erreur augmente.")
            d = imp[imp.target == target].sort_values("importance")
            fig = go.Figure(go.Bar(y=d.feature.map(FEATURE_LABELS).fillna(d.feature), x=d.importance, orientation="h",
                                   marker=dict(color=ui.BLUE, cornerradius=4),
                                   hovertemplate="%{y}<extra></extra>"))
            fig.update_xaxes(title_text="Importance", showticklabels=False)
            ui.plot(ui.style(fig, height=380, legend=False, hover="closest"), key="fc_imp")
