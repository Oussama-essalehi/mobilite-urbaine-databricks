"""Prévision de la fréquentation : prévision 24 h, validation vs baseline, importance des variables."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data, ui

TARGETS = {"Départs": "departures", "Arrivées": "arrivals"}
FEATURE_LABELS = {
    "station_hour_mean": "Moyenne station × heure", "station_mean": "Moyenne station", "hour_of_day": "Heure",
    "lag_24h": "Retard J-1", "lag_168h": "Retard J-7", "weekday": "Jour de semaine", "is_weekend": "Week-end",
    "temperature": "Température", "precipitation": "Précipitations", "wind_speed": "Vent", "capacity": "Capacité",
    "cluster_id": "Segment (cluster)",
}


def render() -> None:
    ui.page_header("Machine learning · supervisé", "Prévision de la fréquentation à 24 h",
                   "Un modèle global par cible (<code>HistGradientBoostingRegressor</code>, perte de Poisson) prévoit "
                   "départs et arrivées de chaque station pour chacune des 24 prochaines heures. Validation temporelle "
                   "sur les 7 derniers jours face à une baseline saisonnière naïve (même heure J-7, sinon J-1).")
    metrics, fc, net = data.load("model_metrics"), data.load("forecast_station_hourly"), data.load("network_hourly")
    if metrics is None or metrics.empty:
        ui.missing("Modèle de prévision (≥ 14 jours d'historique requis)")
        return

    label = st.segmented_control("Cible", list(TARGETS), default="Départs", key="fc_target") or "Départs"
    target = TARGETS[label]
    m = metrics[metrics.target == target].sort_values("trained_at")
    last = m.iloc[-1]
    prev = m.iloc[-2] if len(m) > 1 else None
    ui.kpis([
        ("MAE modèle", ui.fmt_num(last.mae, 3),
         None if prev is None else f"{last.mae - prev.mae:+.3f} vs veille".replace(".", ","),
         "Erreur absolue moyenne, par station et par heure", "inverse"),
        ("MAE baseline", ui.fmt_num(last.baseline_mae, 3), None, "Même heure J-7, sinon J-1"),
        ("Gain", ui.fmt_pct(last.skill_vs_baseline, 1), None, "skill = 1 − MAE modèle / MAE baseline"),
        ("MAE en pointe", ui.fmt_num(last.mae_peak_hours, 3), None, "Semaine, 7-10 h et 17-20 h"),
        ("Entraîné le", f"{pd.Timestamp(last.trained_at):%d/%m}", f"à {pd.Timestamp(last.trained_at):%H:%M}", "Job ML nocturne", "off"),
    ])

    # --- prévision réseau des 24 prochaines heures
    if fc is not None and not fc.empty:
        ui.section("Prochaines 24 heures, tout le réseau", "Prévision comparée à ce qui a été observé à la même heure une semaine plus tôt")
        f = fc.groupby("hour_local")[f"pred_{target}"].sum().reset_index()
        fig = go.Figure()
        if net is not None:
            obs = net.set_index("hour_local")[f"total_{target}"]
            f["j7"] = (f.hour_local - pd.Timedelta(days=7)).map(obs)
            fig.add_scatter(x=f.hour_local, y=f.j7, name="Observé J-7", mode="lines",
                            line=dict(color=ui.MUTED, width=2), hovertemplate="%{y:,.0f}<extra>Observé J-7</extra>")
        fig.add_scatter(x=f.hour_local, y=f[f"pred_{target}"], name="Prévision", mode="lines",
                        line=dict(color=ui.BLUE, width=2.5), fill="tozeroy", fillcolor="rgba(42,120,214,0.10)",
                        hovertemplate="%{y:,.0f}<extra>Prévision</extra>")
        fig.update_yaxes(title_text=f"{label} / heure", rangemode="tozero")
        fig.update_xaxes(tickformat="%d/%m %Hh")
        ui.plot(ui.style(fig, height=340), key="fc_next")

    test, imp = data.load("forecast_test_network"), data.load("feature_importance")
    left, right = st.columns([1.6, 1])
    with left:
        ui.section("Validation : 7 derniers jours", "Somme réseau par heure — réel, modèle et baseline")
        if test is None:
            st.info("Série de test : artefact MLflow `test_reseau_<cible>.png`.", icon=":material/science:")
        else:
            t = test[test.target == target]
            fig = go.Figure()
            for col, name, color, width in [("actual", "Réel", ui.MUTED, 1.6), ("baseline", "Baseline J-7", ui.ORANGE, 1.4),
                                            ("pred", "Modèle", ui.BLUE, 2)]:
                fig.add_scatter(x=t.hour_local, y=t[col], name=name, mode="lines", line=dict(color=color, width=width),
                                hovertemplate="%{y:,.0f}<extra>" + name + "</extra>")
            fig.update_yaxes(title_text=f"{label} / heure")
            fig.update_xaxes(rangeslider=dict(visible=True, thickness=.06), tickformat="%d/%m")
            ui.plot(ui.style(fig, height=380), key="fc_test")
    with right:
        ui.section("Importance des variables", "Hausse de la MAE quand la variable est permutée")
        if imp is None:
            st.info("Importance par permutation : artefact MLflow `importance_<cible>.png`.", icon=":material/science:")
        else:
            d = imp[imp.target == target].sort_values("importance")
            fig = go.Figure(go.Bar(y=d.feature.map(FEATURE_LABELS).fillna(d.feature), x=d.importance, orientation="h",
                                   marker=dict(color=ui.BLUE, cornerradius=4),
                                   hovertemplate="%{y} : +%{x:.3f} MAE<extra></extra>"))
            fig.update_xaxes(title_text="Hausse de MAE")
            ui.plot(ui.style(fig, height=380, legend=False, hover="closest"), key="fc_imp")

    if len(m) > 1:
        ui.section("Suivi des entraînements nocturnes", "Table gold_model_metrics (une ligne par run et par cible)")
        fig = go.Figure()
        fig.add_scatter(x=m.trained_at, y=m.mae, name="Modèle", mode="lines+markers", line=dict(color=ui.BLUE, width=2),
                        marker=dict(size=8), hovertemplate="%{y:.3f}<extra>Modèle</extra>")
        fig.add_scatter(x=m.trained_at, y=m.baseline_mae, name="Baseline J-7", mode="lines+markers",
                        line=dict(color=ui.ORANGE, width=2), marker=dict(size=8), hovertemplate="%{y:.3f}<extra>Baseline</extra>")
        fig.update_yaxes(title_text="MAE", rangemode="tozero")
        fig.update_xaxes(tickformat="%d/%m")
        ui.plot(ui.style(fig, height=280), key="fc_runs")

    if fc is not None and not fc.empty:
        dim = data.load("dim_station")
        ui.section("Prévision d'une station")
        labels = (dim.set_index("station_id").station_label if dim is not None
                  else pd.Series(fc.station_id.unique(), index=fc.station_id.unique()).astype(str))
        busiest = fc.groupby("station_id").pred_departures.sum().sort_values(ascending=False).index
        sid = st.selectbox("Station", busiest, format_func=lambda i: labels.get(i, str(i)), key="fc_station")
        s = fc[fc.station_id == sid].sort_values("hour_local")
        fig = go.Figure()
        fig.add_bar(x=s.hour_local, y=s.pred_departures, name="Départs", marker=dict(color=ui.BLUE, cornerradius=3))
        fig.add_bar(x=s.hour_local, y=s.pred_arrivals, name="Arrivées", marker=dict(color=ui.ORANGE, cornerradius=3))
        fig.update_layout(barmode="group", bargap=.25, bargroupgap=.08)
        fig.update_yaxes(title_text="Vélos / heure")
        fig.update_xaxes(tickformat="%d/%m %Hh")
        ui.plot(ui.style(fig, height=300), key="fc_station_chart")
        net_flow = s.pred_net_flow.sum()
        ui.insight(f"Flux net prévu sur 24 h : <b>{net_flow:+.1f} vélos</b>".replace(".", ",") +
                   (" — la station va plutôt se remplir." if net_flow > 0 else " — la station va plutôt se vider."))

    with st.expander("Méthode", icon=":material/menu_book:"):
        st.markdown("""
- **Variables** : heure, jour, week-end, météo (prévision Open-Meteo à l'inférence), retards J-1 et J-7,
  encodages par station (moyenne globale et par heure, calculés **sur le train uniquement** pour éviter la fuite),
  capacité, segment issu du clustering.
- **Modèle** : `HistGradientBoostingRegressor`, perte de Poisson (adaptée aux comptages), early stopping.
- **Validation** : découpage temporel, 7 derniers jours en test ; MAE, RMSE, MAE aux heures de pointe, gain vs baseline.
- **Suivi** : paramètres, métriques, graphiques et modèle journalisés dans MLflow ; ré-entraînement sur tout l'historique
  avant l'inférence ; enregistrement optionnel dans Unity Catalog.
""")
