"""Tableau de bord Streamlit — Mobilité urbaine à Toulouse (VélôToulouse × météo).

Lancement :  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="Mobilité urbaine · VélôToulouse",
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

ui.inject_css()

nav = st.navigation({
    "Projet": [
        st.Page(home.render, title="Vue d'ensemble", icon=":material/space_dashboard:", default=True),
    ],
    "Analytique": [
        st.Page(realtime.render, title="Disponibilité en direct", icon=":material/location_on:", url_path="temps-reel"),
        st.Page(usage.render, title="Fréquentation", icon=":material/monitoring:", url_path="frequentation"),
        st.Page(weather.render, title="Effet de la météo", icon=":material/rainy:", url_path="meteo"),
        st.Page(saturation.render, title="Stations saturées", icon=":material/warning:", url_path="saturation"),
    ],
    "Machine learning": [
        st.Page(segmentation.render, title="Segmentation des stations", icon=":material/hub:", url_path="segmentation"),
        st.Page(forecast.render, title="Prévision 24 h", icon=":material/query_stats:", url_path="prevision"),
    ],
    "Aide à la décision": [
        st.Page(rebalancing.render, title="Rééquilibrage", icon=":material/local_shipping:", url_path="reequilibrage"),
    ],
    "Ingénierie": [
        st.Page(pipeline.render, title="Pipeline & qualité", icon=":material/account_tree:", url_path="pipeline"),
    ],
})

src = data.source()
with st.sidebar:
    st.markdown("### :material/pedal_bike: Mobilité urbaine")
    st.caption("VélôToulouse × météo · Databricks")
    dot = "#0ca30c" if src.mode == "databricks" else "#eda100"
    st.markdown(f'<div class="src-badge"><b><span class="legend-dot" style="background:{dot}"></span>'
                f'{src.label}</b>{src.detail}</div>', unsafe_allow_html=True)
    if src.error:
        st.warning(f"Connexion Databricks impossible, affichage de la démo.\n\n`{src.error}`")
    st.space("small")
    st.link_button("Code source sur GitHub", REPO, icon=":material/code:", width="stretch")

nav.run()
