"""Pipeline & qualité : jobs, couches médaillon, journal des contrôles qualité (dq_log)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data, ui

JOBS = pd.DataFrame([
    {"Job": "1 · Ingestion", "Planification": "Toutes les 15 min",
     "Tâches": "01 API → landing  →  02 Auto Loader → bronze  →  03 silver (MERGE, expectations, fraîcheur)",
     "Robustesse": "2 reprises, alerte e-mail, échec si source vide"},
    {"Job": "2 · Analytique", "Planification": "Toutes les heures (h+07)",
     "Tâches": "04 gold (7 tables + contrôles)  →  07 recommandations de rééquilibrage",
     "Robustesse": "Contrôles d'unicité, de signe et de couverture météo"},
    {"Job": "3 · Machine learning", "Planification": "Chaque nuit, 4 h 30",
     "Tâches": "05 clustering KMeans  →  06 prévision 24 h (HGB)",
     "Robustesse": "Attente propre si historique insuffisant (7 j / 14 j)"},
])

LAYERS = pd.DataFrame([
    ("Landing", "volume landing/", "Réponse API brute, une par fichier JSON, partitionnée par date"),
    ("Bronze", "bronze_velo_raw, bronze_meteo_raw", "Une ligne par fichier, payload en texte + métadonnées d'ingestion"),
    ("Silver", "silver_station_status", "État de chaque station à chaque collecte (typé, dédoublonné, contrôlé)"),
    ("Silver", "silver_stations · silver_meteo_hourly", "Référentiel des stations (SCD 1) · météo horaire"),
    ("Silver", "silver_velo_quarantine · dq_log", "Lignes rejetées avec règles en échec · historique des contrôles"),
    ("Gold", "gold_station_hourly", "Fait principal station × heure : flux, remplissage, saturation, météo"),
    ("Gold", "gold_network_hourly · gold_dim_station · gold_station_current", "Vue réseau, dimension, temps réel"),
    ("Gold", "gold_saturation_profile · gold_saturation_ranking · gold_weather_effect", "Saturation et effet météo"),
    ("ML", "gold_station_clusters · gold_forecast_station_hourly · gold_model_metrics", "Segments, prévisions, métriques"),
    ("Décision", "gold_rebalancing_latest / _history · gold_rebalancing_moves_*", "Recommandations et mouvements"),
], columns=["Couche", "Tables", "Contenu"])

RULE_LABELS = {
    "station_id_present": "Identifiant présent", "snapshot_ts_present": "Horodatage présent",
    "station_info_found": "Station dans le référentiel", "bikes_non_negative": "Vélos ≥ 0",
    "stands_non_negative": "Bornes ≥ 0", "capacity_positive": "Capacité > 0",
    "occupancy_consistent": "Vélos + bornes ≤ capacité", "coords_in_toulouse": "Coordonnées dans la métropole",
    "time_present": "Heure présente", "temperature_plausible": "Température plausible",
    "precipitation_valid": "Précipitations ≥ 0", "wind_valid": "Vent ≥ 0",
}


def render() -> None:
    ui.page_header("Ingénierie", "Pipeline & qualité des données",
                   "Ingestion planifiée, architecture médaillon sur Delta Lake, traitements incrémentaux et idempotents "
                   "(Auto Loader, Structured Streaming <code>availableNow</code>, MERGE), contrôles qualité déclarés comme "
                   "des expectations SQL, le tout déployé par Databricks Asset Bundle et testé en CI.")

    ui.section("Jobs planifiés", "databricks.yml — compute serverless, une exécution à la fois, alertes sur échec")
    st.table(JOBS.set_index("Job"))

    dq = data.load("dq_log")
    ui.section("Contrôles qualité silver", "Table dq_log : une ligne par règle et par exécution (7 derniers jours)")
    if dq is None or dq.empty:
        ui.missing("Journal qualité")
    else:
        velo = dq[dq.table_name == "silver_station_status"]
        runs = velo.groupby("run_ts").agg(total=("total_rows", "first"), failed=("failed_rows", "sum"),
                                          worst=("failed_rows", "max")).reset_index()
        runs["share"] = runs.worst / runs.total
        incidents = runs[runs.share > .20]
        ui.kpis([
            ("Exécutions", ui.fmt_int(dq.run_ts.nunique()), None, "Collectes des 7 derniers jours"),
            ("Lignes", ui.fmt_int(runs.total.sum()), None, "État des stations, toutes collectes"),
            ("Règles", ui.fmt_int(dq.rule.nunique()), None, "8 règles vélo, 4 règles météo"),
            ("Conformité", ui.fmt_pct(1 - velo.failed_rows.sum() / velo.total_rows.sum(), 2), None,
             "Contrôles réussis / contrôles effectués"),
            ("Lots rejetés", ui.fmt_int(len(incidents)), None, "Plus de 20 % de lignes invalides : le job échoue et alerte"),
        ])
        left, right = st.columns([1, 1.5])
        with left:
            st.markdown("**Taux d'échec par règle**")
            by_rule = dq.groupby("rule").agg(failed=("failed_rows", "sum"), total=("total_rows", "sum")).reset_index()
            by_rule["rate"] = by_rule.failed / by_rule.total
            by_rule = by_rule.sort_values("rate")
            fig = go.Figure(go.Bar(y=by_rule.rule.map(RULE_LABELS).fillna(by_rule.rule), x=by_rule.rate, orientation="h",
                                   marker=dict(color=ui.BLUE, cornerradius=4), customdata=by_rule.failed,
                                   hovertemplate="%{y}<br>%{x:.3%} · %{customdata} lignes<extra></extra>"))
            fig.update_xaxes(tickformat=".2%", title_text="Taux d'échec")
            ui.plot(ui.style(fig, height=400, legend=False, hover="closest"), key="dq_rules")
        with right:
            st.markdown("**Échecs de règles par collecte** (échelle log)")
            fig = go.Figure(go.Scatter(x=runs.run_ts, y=runs.failed, mode="lines", line=dict(color=ui.BLUE, width=1.5),
                                       hovertemplate="%{x|%d/%m %H:%M} : %{y} échecs<extra></extra>", name="Échecs"))
            if not incidents.empty:
                fig.add_scatter(x=incidents.run_ts, y=incidents.failed, mode="markers", name="Lot rejeté",
                                marker=dict(color="#d03b3b", size=11, line=dict(width=2, color="white")),
                                hovertemplate="%{x|%d/%m %H:%M} : lot rejeté<extra></extra>")
                r = incidents.iloc[0]
                fig.add_annotation(x=r.run_ts, y=r.failed, text="Flux station_information incomplet → quarantaine + alerte",
                                   showarrow=True, arrowhead=0, ax=-40, ay=-10, xanchor="right",
                                   font=dict(size=11, color=ui.MUTED))
            fig.update_yaxes(type="log", dtick=1, title_text="Échecs")
            fig.update_xaxes(tickformat="%d/%m")
            ui.plot(ui.style(fig, height=400, legend=False, hover="closest"), key="dq_runs")
        ui.insight("Une ligne invalide part en <b>quarantaine</b> avec la liste des règles en échec au lieu de faire "
                   "planter le pipeline ; au-delà de 20 % d'un lot, ou si la dernière collecte a plus d'une heure, la "
                   "tâche échoue et le job envoie une alerte. Rejouer un fichier ne crée pas de doublon (MERGE).")

    ui.section("Couches de données", "Catalogue Unity Catalog workspace.mobility")
    st.dataframe(LAYERS, hide_index=True, width="stretch")

    c1, c2, c3 = st.columns(3)
    with c1.container(border=True, height="stretch"):
        st.markdown("**:material/verified: Tests & CI**")
        st.caption("pytest sur la logique pure (rééquilibrage, fenêtres de pointe, encodages, libellés de clusters), "
                   "ruff, validation du bundle — GitHub Actions à chaque push.")
    with c2.container(border=True, height="stretch"):
        st.markdown("**:material/schema: Choix de modélisation**")
        st.caption("Bronze en texte brut : un changement de schéma de l'API se traduit par de la quarantaine, jamais "
                   "par un échec d'ingestion. Gold en clustering liquide (station, heure) pour Power BI en DirectQuery.")
    with c3.container(border=True, height="stretch"):
        st.markdown("**:material/bar_chart: Restitution**")
        st.caption("Power BI en DirectQuery sur le SQL warehouse (mesures DAX dans powerbi/) et cette application "
                   "Streamlit, qui lit les mêmes tables gold.")
