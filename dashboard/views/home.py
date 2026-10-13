"""Accueil : le projet en quelques mots, les chiffres clés et les grands enseignements."""
import pandas as pd
import streamlit as st

from lib import data, ui

STEPS = [
    ("#3987e5", "Collecter", "Toutes les 15 minutes, le nombre de vélos et de places libres de chaque station, "
                             "et la météo de Toulouse."),
    ("#c98500", "Vérifier et ranger", "Les données incohérentes sont écartées automatiquement, le reste est "
                                      "organisé pour être analysé rapidement."),
    ("#199e70", "Analyser", "Heures d'affluence, effet de la pluie, stations souvent vides ou pleines, familles "
                            "de stations."),
    ("#d95926", "Prévoir et conseiller", "Affluence des 24 prochaines heures et vélos à déplacer avant chaque "
                                         "heure de pointe."),
]

PAGES = [
    (":material/location_on:", "Les vélos en ce moment", "La carte de toutes les stations : où trouver un vélo, où le rendre."),
    (":material/schedule:", "Quand roule-t-on ?", "Les jours et les heures où les vélos sont le plus utilisés."),
    (":material/rainy:", "L'effet de la météo", "Combien de trajets en moins quand il pleut."),
    (":material/warning:", "Stations à problème", "Les stations souvent vides ou pleines aux heures de pointe."),
    (":material/category:", "Familles de stations", "Les stations regroupées selon leur comportement dans la journée."),
    (":material/insights:", "Prévisions 24 heures", "Combien de vélos vont partir et arriver dans les prochaines heures."),
    (":material/local_shipping:", "Déplacer les vélos", "Le plan conseillé pour remettre des vélos là où il en manquera."),
    (":material/settings:", "Comment ça marche", "Les coulisses : collecte automatique et contrôles de qualité."),
]


def render() -> None:
    st.markdown(
        """<div class="hero"><div class="page-eyebrow">VéloScope Toulouse · Projet de données</div>
        <h1>Les vélos en libre-service de Toulouse, analysés automatiquement</h1>
        <p>Ce projet observe en continu les stations VélôToulouse et la météo, puis transforme ces données en réponses
        simples : quand et où les Toulousains prennent le vélo, ce que change la pluie, quelles stations manquent de
        vélos ou de places, et comment mieux répartir les vélos avant les heures de pointe.</p></div>""",
        unsafe_allow_html=True)

    dim, net, cur = data.load("dim_station"), data.load("network_hourly"), data.load("station_current")
    metrics, effect, ranking = data.load("model_metrics"), data.load("weather_effect"), data.load("saturation_ranking")

    items = []
    if dim is not None:
        items.append(("Stations observées", ui.fmt_int(len(dim)), "dans la métropole", None))
    if net is not None and not net.empty:
        today = net.date_local.max()
        daily = net[net.date_local < today].groupby("date_local").total_departures.sum()
        items.append(("Trajets par jour", ui.fmt_int(daily.tail(7).mean()), "moyenne des 7 derniers jours",
                      "Estimation à partir des vélos qui quittent les stations"))
    if cur is not None and not cur.empty:
        share = (cur[cur.status == "OPEN"].availability_status == "Disponible").mean()
        items.append(("Stations sans souci", ui.fmt_pct(share), "à la dernière mesure",
                      "Stations ni vides, ni pleines, ni sur le point de l'être"))
    if metrics is not None and not metrics.empty:
        last = metrics[metrics.target == "departures"].sort_values("trained_at").iloc[-1]
        items.append(("Prévisions plus justes", ui.fmt_pct(last.skill_vs_baseline), "qu'une méthode simple",
                      "Comparé à « même heure, la semaine précédente »"))
    if items:
        ui.kpis(items)

    ui.section("Comment ça marche, en 4 étapes", "Tout se fait automatiquement, jour et nuit.")
    steps = "".join(f'<div class="step"><span class="n" style="background:{c}">{i}</span><div class="t">{t}</div>'
                    f'<div class="d">{d}</div></div>' for i, (c, t, d) in enumerate(STEPS, 1))
    st.markdown(f'<div class="steps">{steps}</div>', unsafe_allow_html=True)

    ui.section("Ce que disent les données")
    cards = []
    if effect is not None and not effect.empty:
        rain = effect[(effect.dimension == "Météo") & (effect.category == "Pluie")]
        if not rain.empty:
            cards.append(("La pluie", ui.fmt_pct(rain.relative_to_reference.iloc[0] - 1),
                          "de trajets quand il pleut, par rapport à une heure sans pluie."))
    if net is not None and not net.empty:
        wd = net[net.day_type == "Semaine"].groupby("hour_of_day").total_departures.mean()
        cards.append(("Le pic de la journée", f"{wd.idxmax()} h",
                      f"En semaine, c'est l'heure où l'on prend le plus de vélos : environ {ui.fmt_int(wd.max())} "
                      "trajets en une heure."))
    if ranking is not None and not ranking.empty:
        n = int((ranking.pct_saturated_peak >= .25).sum())
        cards.append(("Les stations à problème", f"{n}",
                      "stations sont vides ou pleines plus d'un quart du temps aux heures de pointe."))
    if metrics is not None and not metrics.empty:
        last = metrics[metrics.target == "departures"].sort_values("trained_at").iloc[-1]
        cards.append(("La prévision", f"± {ui.fmt_num(last.mae, 1)} vélo",
                      "d'erreur en moyenne, par station et par heure, sur les départs prévus."))
    if cards:
        cols = st.columns(len(cards))
        for col, (k, v, d) in zip(cols, cards):
            col.markdown(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{d}</div></div>',
                         unsafe_allow_html=True)

    ui.section("Que trouver sur ce site ?", "Le menu à gauche donne accès à chaque page.")
    for row in range(0, len(PAGES), 4):
        cols = st.columns(4)
        for col, (icon, title, desc) in zip(cols, PAGES[row:row + 4]):
            with col.container(border=True, height="stretch"):
                st.markdown(f"**{icon} {title}**")
                st.caption(desc)

    st.space("small")
    if data.source().mode == "demo":
        st.caption(":material/info: **À propos des données affichées.** Les stations (noms, emplacements, nombre de "
                   "places) sont réelles. L'historique est simulé de façon réaliste pour la démonstration, puis traité "
                   "exactement comme les vraies données. La page *Les vélos en ce moment* montre l'état réel, en direct.")
    st.caption(f"Analyses calculées le {pd.Timestamp(data.reference_time()):%d/%m/%Y à %H:%M}.")
