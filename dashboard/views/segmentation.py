"""Familles de stations : stations regroupées automatiquement selon leur comportement dans la journée."""
import re

import plotly.graph_objects as go
import streamlit as st

from lib import data, ui

# nom et explication grand public de chaque type de famille (libellés produits par src/mobility/ml_utils.py)
FAMILIES = {
    "Résidentielle": ("Quartiers résidentiels",
                      "Pleines la nuit, elles se vident le matin quand les habitants partent travailler et se "
                      "remplissent le soir à leur retour."),
    "Pôle d'activité": ("Lieux de travail et d'études",
                        "Presque vides la nuit, elles se remplissent le matin (bureaux, universités) et se vident "
                        "le soir."),
    "Équilibrée": ("Centre-ville et loisirs",
                   "Remplissage assez stable : les vélos partent et arrivent tout au long de la journée."),
    "Souvent vide": ("Souvent vides", "Il y manque presque toujours des vélos : ce sont des stations à surveiller."),
    "Souvent pleine": ("Souvent pleines", "Il est souvent difficile d'y rendre un vélo faute de place."),
}


def family(label: str) -> tuple[str, str]:
    for key, (name, desc) in FAMILIES.items():
        if label.startswith(key):
            n = re.search(r" \((\d+)\)$", label)       # libellés dédoublonnés : « … (2) »
            return name + (f" {n.group(1)}" if n else ""), desc
    return label, ""


def render() -> None:
    ui.page_header("Prévoir", "Familles de stations",
                   "Un programme a comparé, heure par heure, la façon dont chaque station se remplit et se vide. "
                   "Il a ensuite regroupé automatiquement les stations qui se comportent de la même manière.")
    clusters, dim, profiles = data.load("station_clusters"), data.load("dim_station"), data.load("cluster_profiles")
    if clusters is None or clusters.empty:
        ui.missing("Les familles de stations (il faut au moins 7 jours de données)")
        return

    colors = ui.cluster_colors(clusters.cluster_label)
    names = {lab: family(lab)[0] for lab in colors}
    name_colors = {names[lab]: c for lab, c in colors.items()}
    counts = clusters.cluster_label.value_counts()

    ui.section(f"{len(colors)} familles trouvées")
    cols = st.columns(len(colors))
    for col, lab in zip(cols, sorted(colors, key=lambda x: -counts.get(x, 0))):
        name, desc = family(lab)
        col.markdown(f'<div class="card"><div class="k" style="color:{colors[lab]}">'
                     f'<span class="legend-dot" style="background:{colors[lab]}"></span>{counts.get(lab, 0)} stations</div>'
                     f'<div class="t" style="font-weight:650;margin:4px 0">{name}</div><div class="d">{desc}</div></div>',
                     unsafe_allow_html=True)

    if profiles is not None and not profiles.empty:
        ui.section("Comment chaque famille se remplit au fil de la journée", "0 % = station vide, 100 % = station pleine")
        dt = st.segmented_control("Jour", ["Semaine", "Week-end"], default="Semaine", key="seg_day") or "Semaine"
        fig = go.Figure()
        for lab in sorted(colors, key=lambda x: -counts.get(x, 0)):
            d = profiles[(profiles.cluster_label == lab) & (profiles.day_type == dt)].sort_values("hour_of_day")
            if d.empty:
                continue
            fig.add_scatter(x=d.hour_of_day, y=d.avg_fill_rate, mode="lines", name=names[lab],
                            line=dict(color=colors[lab], width=2.5), hovertemplate="%{y:.0%}<extra>" + names[lab] + "</extra>")
        fig.update_yaxes(title_text="Remplissage", tickformat=".0%", range=[0, 1])
        fig.update_xaxes(title_text="Heure de la journée", dtick=2, ticksuffix=" h")
        ui.plot(ui.style(fig, height=400), key="seg_profiles")
        ui.insight("Le matin, les stations des <b>quartiers résidentiels</b> se vident pendant que celles des "
                   "<b>lieux de travail et d'études</b> se remplissent : c'est le trajet domicile-travail. Le soir, "
                   "le mouvement s'inverse. Le week-end, ces différences s'effacent.")

    if dim is not None:
        ui.section("Où se trouvent ces familles ?")
        mp = dim.drop(columns=["cluster_id", "cluster_label"], errors="ignore").merge(
            clusters[["station_id", "cluster_label"]], on="station_id")
        mp["famille"] = mp.cluster_label.map(names)
        fig = ui.station_map(mp, "famille", name_colors, height=500,
                             hover={"famille": False, "lat": False, "lon": False})
        ui.plot(fig, key="seg_map")

    sel = data.load("cluster_k_selection")
    with st.expander("Pour les curieux : comment le nombre de familles est choisi", icon=":material/lightbulb:"):
        st.markdown("Le programme a essayé de former entre 3 et 8 familles. Pour chaque essai, il mesure à quel point "
                    "les familles sont bien séparées les unes des autres (score de 0 à 1, plus c'est haut, mieux "
                    f"c'est). Le meilleur score est obtenu avec **{clusters.k.iloc[0]} familles** "
                    f"(score {ui.fmt_num(clusters.silhouette.iloc[0])}).")
        if sel is not None:
            best = int(clusters.k.iloc[0])
            fig = go.Figure(go.Scatter(x=sel.k, y=sel.silhouette, mode="lines+markers", line=dict(color=ui.BLUE, width=2),
                                       marker=dict(size=9, color=[ui.ORANGE if k == best else ui.BLUE for k in sel.k]),
                                       hovertemplate="%{x} familles : score %{y:.2f}<extra></extra>"))
            fig.update_xaxes(title_text="Nombre de familles essayé", dtick=1)
            fig.update_yaxes(title_text="Score de séparation")
            ui.plot(ui.style(fig, height=280, legend=False, hover="closest"), key="seg_k")
