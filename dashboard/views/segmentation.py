"""Segmentation des stations : profils de remplissage par cluster, carte, sélection de k, ACP."""
import plotly.graph_objects as go
import streamlit as st

from lib import data, ui

RULES = """
| Règle (dans l'ordre) | Libellé |
|---|---|
| vide ≥ 25 % du temps | **Souvent vide** |
| pleine ≥ 25 % du temps | **Souvent pleine** |
| remplissage 5-7 h − 10-16 h ≥ 15 pts | **Résidentielle** (se vide le matin) |
| remplissage 10-16 h − 5-7 h ≥ 15 pts | **Pôle d'activité** (se remplit le matin) |
| sinon | **Équilibrée / loisirs** |
"""


def render() -> None:
    ui.page_header("Machine learning · non supervisé", "Segmentation des stations",
                   "Chaque station est décrite par son profil horaire de remplissage (24 h semaine + 24 h week-end) et "
                   "sa fréquence de saturation. Un KMeans est entraîné pour k = 3…8, le meilleur k est choisi par "
                   "silhouette, chaque essai est suivi dans MLflow, puis chaque cluster reçoit un libellé métier.")
    clusters, dim, profiles = data.load("station_clusters"), data.load("dim_station"), data.load("cluster_profiles")
    if clusters is None or clusters.empty:
        ui.missing("Clustering (≥ 7 jours d'historique requis)")
        return
    colors = ui.cluster_colors(clusters.cluster_label)
    counts = clusters.cluster_label.value_counts()

    ui.kpis([
        ("Nombre de segments (k)", f"{clusters.k.iloc[0]}", None, "Retenu par le score de silhouette"),
        ("Silhouette", ui.fmt_num(clusters.silhouette.iloc[0]), None, "De −1 à 1 : séparation des clusters"),
        ("Stations segmentées", ui.fmt_int(len(clusters)), None, None),
        ("Segment majoritaire", counts.index[0].split(" (")[0], None, f"{counts.iloc[0]} stations"),
    ])

    if profiles is not None and not profiles.empty:
        ui.section("Signature horaire de chaque segment", "Taux de remplissage moyen par heure")
        dt = st.segmented_control("Type de jour", ["Semaine", "Week-end"], default="Semaine", key="seg_day") or "Semaine"
        fig = go.Figure()
        for lab in colors:
            d = profiles[(profiles.cluster_label == lab) & (profiles.day_type == dt)].sort_values("hour_of_day")
            if d.empty:
                continue
            fig.add_scatter(x=d.hour_of_day, y=d.avg_fill_rate, mode="lines", name=f"{lab} · {counts.get(lab, 0)}",
                            line=dict(color=colors[lab], width=2.5), hovertemplate="%{y:.0%}<extra>" + lab + "</extra>")
        fig.update_yaxes(title_text="Taux de remplissage", tickformat=".0%", range=[0, 1])
        fig.update_xaxes(title_text="Heure", dtick=2, ticksuffix=" h")
        ui.plot(ui.style(fig, height=400), key="seg_profiles")
        ui.insight("Les stations <b>résidentielles</b> se vident entre 7 h et 9 h quand les pôles d'activité se "
                   "remplissent : c'est le mouvement pendulaire domicile-travail, qui s'inverse le soir. Le week-end, "
                   "ces signatures s'aplatissent.")

    left, right = st.columns([1.4, 1])
    with left:
        ui.section("Répartition géographique")
        if dim is not None:
            mp = dim.drop(columns=["cluster_id", "cluster_label"], errors="ignore").merge(
                clusters[["station_id", "cluster_label"]], on="station_id")
            fig = ui.station_map(mp, "cluster_label", colors, height=470,
                                 hover={"capacity": True, "cluster_label": False, "lat": False, "lon": False})
            ui.plot(fig, key="seg_map")
    with right:
        ui.section("Règles de nommage", "Appliquées au profil moyen de chaque cluster (src/mobility/ml_utils.py)")
        st.markdown(RULES)
        summary = data.load("cluster_summary")
        if summary is not None:
            pct = ["early_fill", "midday_fill", "pct_empty", "pct_full"]
            summary = summary.assign(cluster_label=summary.cluster_label.str.split(" (", regex=False).str[0])
            summary[pct] = summary[pct] * 100
            st.dataframe(summary[["cluster_label", "n_stations", "early_fill", "midday_fill", "pct_empty", "pct_full"]],
                         hide_index=True, width="stretch", column_config={
                             "cluster_label": "Segment", "n_stations": st.column_config.NumberColumn("N", width=40),
                             "early_fill": st.column_config.NumberColumn("5-7 h", format="%.0f%%"),
                             "midday_fill": st.column_config.NumberColumn("10-16 h", format="%.0f%%"),
                             "pct_empty": st.column_config.NumberColumn("Vide", format="%.0f%%"),
                             "pct_full": st.column_config.NumberColumn("Pleine", format="%.0f%%")})

    sel, pca = data.load("cluster_k_selection"), data.load("cluster_pca")
    ui.section("Sélection du modèle", "Artefacts MLflow de l'entraînement")
    if sel is None or pca is None:
        st.info("Graphiques de sélection de k et projection ACP : disponibles dans l'expérience MLflow "
                "`mobilite_urbaine` (artefacts `selection_k.png`, `acp_clusters.png`).", icon=":material/science:")
        return
    best = int(clusters.k.iloc[0])
    c1, c2, c3 = st.columns(3)
    for col, metric, title, fmt in [(c1, "silhouette", "Silhouette selon k (à maximiser)", ".2f"),
                                    (c2, "inertia", "Inertie selon k (coude)", ",.0f")]:
        with col:
            st.markdown(f"**{title}**")
            fig = go.Figure(go.Scatter(x=sel.k, y=sel[metric], mode="lines+markers", line=dict(color=ui.BLUE, width=2),
                                       marker=dict(size=8, color=[ui.ORANGE if k == best else ui.BLUE for k in sel.k]),
                                       hovertemplate="k = %{x} : %{y:" + fmt + "}<extra></extra>"))
            fig.update_xaxes(title_text="k", dtick=1)
            ui.plot(ui.style(fig, height=300, legend=False, hover="closest"), key=f"seg_{metric}")
    with c3:
        st.markdown(f"**Projection ACP (k = {best})**")
        pts = pca.merge(clusters[["station_id", "cluster_label"]], on="station_id")
        if dim is not None:
            pts = pts.merge(dim[["station_id", "station_label"]], on="station_id", how="left")
        fig = go.Figure()
        for lab, color in colors.items():
            d = pts[pts.cluster_label == lab]
            fig.add_scatter(x=d.pc1, y=d.pc2, mode="markers", name=lab, showlegend=False,
                            marker=dict(color=color, size=7, opacity=.85, line=dict(width=1, color="white")),
                            text=d.get("station_label"), hovertemplate="%{text}<extra>" + lab + "</extra>")
        fig.update_xaxes(title_text="CP1", showticklabels=False)
        fig.update_yaxes(title_text="CP2", showticklabels=False)
        ui.plot(ui.style(fig, height=300, legend=False, hover="closest"), key="seg_pca")
    ui.data_table(sel.round(3), "Scores par valeur de k")
