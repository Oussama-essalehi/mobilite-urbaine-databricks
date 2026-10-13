"""Tableau de bord Streamlit — Les vélos en libre-service de Toulouse (VélôToulouse × météo).

Lancement :  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="Vélos en libre-service à Toulouse",
    page_icon=":material/pedal_bike:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from lib import data, ui  # noqa: E402
from views import (  # noqa: E402
    forecast,
    home,
    pipeline,
    realtime,
    rebalancing,
    saturation,
    segmentation,
    usage,
    weather,
)

REPO = "https://github.com/Oussama-essalehi/mobilite-urbaine-databricks"


def _mark_seen() -> None:
    st.session_state.welcome_seen = True


@st.dialog("Bienvenue", width="large", icon=":material/waving_hand:", on_dismiss=_mark_seen)
def welcome() -> None:
    st.markdown(
        """
### Les vélos en libre-service de Toulouse, analysés automatiquement

**Le point de départ.** Toulouse compte plus de 460 stations VélôToulouse. Un problème revient tous les jours :
certaines stations sont **vides** quand on veut prendre un vélo, d'autres sont **pleines** quand on veut le rendre.

**Ce que fait ce projet.**

1. **Il collecte** toutes les 15 minutes, sans intervention humaine, le nombre de vélos et de places libres dans
   chaque station, ainsi que la météo.
2. **Il vérifie et range** ces données : les informations incohérentes sont écartées automatiquement.
3. **Il analyse** : à quelles heures on roule le plus, l'effet de la pluie, les stations qui posent problème.
4. **Il prévoit et conseille** : il estime l'affluence des 24 prochaines heures et propose où déplacer des vélos
   avant les heures de pointe.

**Comment naviguer.** Le menu à gauche est organisé en quatre parties : *Comprendre*, *Prévoir*, *Agir* et
*Coulisses*. Chaque page commence par une phrase qui explique ce qu'elle montre, et se termine par ce qu'il faut
en retenir.
"""
    )
    src = data.source()
    if src.mode == "demo":
        st.info("Pour cette démonstration, les stations sont réelles mais l'historique est **simulé**. Seule la page "
                "*Les vélos en ce moment* affiche l'état réel des stations, en direct.", icon=":material/info:")
    st.caption("Projet réalisé par Oussama Essalehi · Outils : Databricks, Python, Power BI, Streamlit")
    if st.button("Découvrir le tableau de bord", type="primary", width="stretch"):
        st.session_state.welcome_seen = True
        st.rerun()


ui.inject_css()

nav = st.navigation({
    "Le projet": [
        st.Page(home.render, title="Accueil", icon=":material/home:", default=True),
    ],
    "Comprendre": [
        st.Page(realtime.render, title="Les vélos en ce moment", icon=":material/location_on:", url_path="en-direct"),
        st.Page(usage.render, title="Quand roule-t-on ?", icon=":material/schedule:", url_path="utilisation"),
        st.Page(weather.render, title="L'effet de la météo", icon=":material/rainy:", url_path="meteo"),
        st.Page(saturation.render, title="Stations à problème", icon=":material/warning:", url_path="stations-a-probleme"),
    ],
    "Prévoir": [
        st.Page(segmentation.render, title="Familles de stations", icon=":material/category:", url_path="familles"),
        st.Page(forecast.render, title="Prévisions 24 heures", icon=":material/insights:", url_path="previsions"),
    ],
    "Agir": [
        st.Page(rebalancing.render, title="Déplacer les vélos", icon=":material/local_shipping:", url_path="redistribution"),
    ],
    "Coulisses": [
        st.Page(pipeline.render, title="Comment ça marche", icon=":material/settings:", url_path="coulisses"),
    ],
})

src = data.source()
with st.sidebar:
    st.markdown("### :material/pedal_bike: Vélos à Toulouse")
    dot = "#0ca30c" if src.mode == "databricks" else "#c98500"
    st.markdown(f'<div class="src-badge"><b><span class="legend-dot" style="background:{dot}"></span>'
                f'{src.label}</b>{src.detail}</div>', unsafe_allow_html=True)
    if src.error:
        st.warning("Impossible de se connecter à Databricks : affichage des données de démonstration. "
                   "Vérifiez les identifiants dans le fichier des secrets.", icon=":material/cloud_off:")
    st.space("small")
    if st.button("Présentation du projet", icon=":material/info:", width="stretch"):
        st.session_state.welcome_seen = False
    st.link_button("Voir le code source", REPO, icon=":material/code:", width="stretch")

if not st.session_state.get("welcome_seen", False):
    welcome()

nav.run()
