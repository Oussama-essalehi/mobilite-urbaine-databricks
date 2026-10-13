"""Vue d'ensemble : pitch du projet, chiffres clés, architecture, résultats."""
import pandas as pd
import streamlit as st

from lib import data, ui

STACK = ["Databricks serverless", "PySpark", "Spark SQL", "Delta Lake", "Auto Loader", "Unity Catalog", "MLflow",
         "scikit-learn", "Databricks Asset Bundles", "GitHub Actions", "Power BI DirectQuery", "Streamlit"]

FLOW = [
    ("Sources", "#2a78d6", "API temps réel", "GBFS VélôToulouse + Open-Meteo, toutes les 15 min, reprises avec backoff"),
    ("Landing", "#6250d6", "JSON brut", "Une réponse par fichier, volume Unity Catalog partitionné par date"),
    ("Bronze", "#eb6834", "Auto Loader", "Ingestion incrémentale, payload conservé en texte : robuste aux changements de schéma"),
    ("Silver", "#898781", "Qualité intégrée", "MERGE idempotent, 12 expectations SQL, quarantaine, contrôle de fraîcheur"),
    ("Gold", "#eda100", "Modèle métier", "Faits station × heure, saturation, effet météo, disponibilité courante"),
    ("ML", "#1baf7a", "MLflow", "Segmentation KMeans et prévision 24 h par gradient boosting, suivies et versionnées"),
    ("Décision", "#e34948", "Rééquilibrage", "Stations à réapprovisionner / délester avant chaque pointe, tournées ≤ 3 km"),
]

PAGES = [
    (":material/location_on:", "Disponibilité en direct", "Carte des 460+ stations : vélos, bornes, stations vides ou pleines."),
    (":material/monitoring:", "Fréquentation", "Départs estimés, profils semaine / week-end, carte de chaleur, top stations."),
    (":material/rainy:", "Effet de la météo", "Impact de la pluie et de la température à heure comparable."),
    (":material/warning:", "Stations saturées", "Classement aux heures de pointe et profil horaire de chaque station."),
    (":material/hub:", "Segmentation", "Typologie des stations par KMeans : résidentielles, pôles d'activité…"),
    (":material/query_stats:", "Prévision 24 h", "Départs et arrivées par station, gain vs baseline, importance des variables."),
    (":material/local_shipping:", "Rééquilibrage", "Plan d'action avant la prochaine pointe et mouvements proposés."),
    (":material/account_tree:", "Pipeline & qualité", "Jobs planifiés, couches médaillon, journal des contrôles qualité."),
]


def render() -> None:
    src = data.source()
    tags = "".join(f'<span class="tag">{t}</span>' for t in STACK)
    st.markdown(
        f"""<div class="hero"><div class="page-eyebrow">Data engineering · Machine learning · Aide à la décision</div>
        <h1>Mobilité urbaine à Toulouse</h1>
        <p>Pipeline de données de bout en bout sur <b>Databricks</b> autour des vélos en libre-service
        <b>VélôToulouse</b> et de la météo : collecte temps réel, architecture médaillon sur Delta Lake avec contrôles
        qualité, segmentation des stations et prévision de la fréquentation suivies dans MLflow, puis recommandations
        de rééquilibrage avant chaque heure de pointe.</p>{tags}</div>""",
        unsafe_allow_html=True)

    dim, net, cur = data.load("dim_station"), data.load("network_hourly"), data.load("station_current")
    metrics, clusters, effect = data.load("model_metrics"), data.load("station_clusters"), data.load("weather_effect")

    items = []
    if dim is not None:
        items.append(("Stations suivies", ui.fmt_int(len(dim)), None, "Référentiel issu du flux GBFS (silver_stations)"))
    if net is not None and not net.empty:
        today = net.date_local.max()
        daily = net[net.date_local < today].groupby("date_local").total_departures.sum()
        items.append(("Départs / jour", ui.fmt_int(daily.tail(7).mean()), None,
                      "Moyenne des 7 derniers jours complets. Estimation par variation de stock (minorant)."))
    if cur is not None and not cur.empty:
        open_ = cur[cur.status == "OPEN"]
        share = (open_.availability_status == "Disponible").mean()
        items.append(("Stations OK", ui.fmt_pct(share), None, "Part des stations ouvertes ni vides ni pleines"))
    if metrics is not None and not metrics.empty:
        last = metrics[metrics.target == "departures"].sort_values("trained_at").iloc[-1]
        items.append(("Gain du modèle", ui.fmt_pct(last.skill_vs_baseline), None,
                      "Réduction de la MAE des départs prévus par rapport à la baseline saisonnière (J-7)"))
    if clusters is not None and not clusters.empty:
        items.append(("Segments", f"{clusters.k.iloc[0]}", None,
                      f"KMeans, silhouette {ui.fmt_num(clusters.silhouette.iloc[0])}"))
    if items:
        ui.kpis(items)

    ui.section("Architecture du pipeline", "Trois jobs planifiés déployés par Databricks Asset Bundle : collecte "
               "(15 min), analytique (horaire), machine learning (chaque nuit).")
    steps = "".join(
        f'<div class="step"><div class="k" style="color:{c}">{i} · {k}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for i, (k, c, t, d) in enumerate(FLOW, 1))
    st.markdown(f'<div class="flow">{steps}</div>', unsafe_allow_html=True)

    ui.section("Ce que disent les données")
    cards = []
    if effect is not None and not effect.empty:
        rain = effect[(effect.dimension == "Météo") & (effect.category == "Pluie")]
        if not rain.empty:
            cards.append(("Pluie", ui.fmt_pct(rain.relative_to_reference.iloc[0] - 1, signed=True),
                          "de départs sous la pluie, à heure et type de jour comparables."))
    if net is not None and not net.empty:
        wd = net[net.day_type == "Semaine"].groupby("hour_of_day").total_departures.mean()
        cards.append(("Pic de demande", f"{wd.idxmax()} h",
                      f"en semaine : {ui.fmt_int(wd.max())} départs/h en moyenne, {wd.max() / wd.mean():.1f}× la moyenne horaire."
                      .replace(".", ",", 1)))
    ranking = data.load("saturation_ranking")
    if ranking is not None and not ranking.empty:
        n = int((ranking.pct_saturated_peak >= .25).sum())
        cards.append(("Saturation", f"{n} stations",
                      "vides ou pleines plus d'un quart du temps aux heures de pointe de semaine."))
    if metrics is not None and not metrics.empty:
        last = metrics[metrics.target == "departures"].sort_values("trained_at").iloc[-1]
        cards.append(("Prévision", f"MAE {ui.fmt_num(last.mae)}",
                      f"départ par station et par heure, contre {ui.fmt_num(last.baseline_mae)} pour la baseline J-7."))
    if cards:
        cols = st.columns(len(cards))
        for col, (k, v, d) in zip(cols, cards):
            col.markdown(f'<div class="card"><div class="page-eyebrow">{k}</div>'
                         f'<div class="page-title" style="font-size:1.6rem">{v}</div><div class="d">{d}</div></div>',
                         unsafe_allow_html=True)

    ui.section("Explorer le tableau de bord", "Chaque page reprend une table gold du pipeline.")
    for row in range(0, len(PAGES), 4):
        cols = st.columns(4)
        for col, (icon, title, desc) in zip(cols, PAGES[row:row + 4]):
            with col.container(border=True, height="stretch"):
                st.markdown(f"**{icon} {title}**")
                st.caption(desc)

    st.space("small")
    if src.mode == "demo":
        meta = data.demo_meta()
        st.caption(
            f":material/info: **Données de démonstration.** Les {meta.get('n_stations', '')} stations (noms, positions, "
            "capacités) proviennent du flux GBFS public ; l'historique horaire est simulé de façon réaliste, puis "
            "traité par le même code que le pipeline (tables gold, KMeans, gradient boosting, rééquilibrage). "
            "La page *Disponibilité en direct* interroge l'API réelle. Avec des identifiants Databricks, "
            "l'application lit directement les tables gold du SQL warehouse.")
    st.caption(f"Instant de référence des analyses : {pd.Timestamp(data.reference_time()):%d/%m/%Y %H:%M} (heure de Paris).")
