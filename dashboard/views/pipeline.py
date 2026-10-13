"""Comment ça marche : la collecte automatique, les contrôles de qualité et les outils utilisés."""
import plotly.graph_objects as go
import streamlit as st

from lib import data, ui

STEPS = [
    ("#3987e5", "Toutes les 15 minutes", "Récupération de l'état de chaque station auprès de VélôToulouse et de la "
                                         "météo, puis vérification et rangement des données."),
    ("#c98500", "Toutes les heures", "Calcul des indicateurs (trajets, stations vides ou pleines, effet de la "
                                     "météo) et du plan de déplacement des vélos."),
    ("#199e70", "Chaque nuit", "Mise à jour des familles de stations et nouvel apprentissage du modèle de prévision."),
    ("#d95926", "En cas de problème", "Si une source ne répond pas ou si les données sont anormales, le traitement "
                                      "s'arrête et une alerte est envoyée par courriel."),
]

TOOLS = [
    (":material/cloud:", "Databricks", "Plateforme en ligne qui exécute les traitements automatiques et stocke les données."),
    (":material/code:", "Python", "Langage de programmation utilisé pour les traitements, les modèles et ce site."),
    (":material/bar_chart:", "Power BI", "Logiciel de tableaux de bord, branché directement sur les données."),
    (":material/web:", "Streamlit", "Outil qui permet de construire ce site à partir du code Python."),
]


def render() -> None:
    ui.page_header("Coulisses", "Comment ça marche",
                   "Personne n'a besoin d'intervenir : la collecte, les vérifications, les calculs et les prévisions "
                   "se lancent tout seuls, à intervalles réguliers.")

    ui.section("Un fonctionnement automatique")
    steps = "".join(f'<div class="step"><span class="n" style="background:{c}">{i}</span><div class="t">{t}</div>'
                    f'<div class="d">{d}</div></div>' for i, (c, t, d) in enumerate(STEPS, 1))
    st.markdown(f'<div class="steps">{steps}</div>', unsafe_allow_html=True)

    dq = data.load("dq_log")
    ui.section("Des données vérifiées à chaque collecte",
               "Chaque mesure passe une série de contrôles : un nombre de vélos négatif, une station inconnue ou "
               "des coordonnées hors de Toulouse sont écartés automatiquement.")
    if dq is None or dq.empty:
        ui.missing("Le journal des contrôles")
    else:
        velo = dq[dq.table_name == "silver_station_status"]
        runs = velo.groupby("run_ts").agg(total=("total_rows", "first"), failed=("failed_rows", "sum"),
                                          worst=("failed_rows", "max")).reset_index()
        alerts = runs[runs.worst / runs.total > .20]
        ui.kpis([
            ("Collectes vérifiées", ui.fmt_int(dq.run_ts.nunique()), "sur les 7 derniers jours", None),
            ("Contrôles par mesure", ui.fmt_int(dq.rule.nunique()), "vérifications différentes", None),
            ("Données conformes", ui.fmt_pct(1 - velo.failed_rows.sum() / velo.total_rows.sum(), 2), "des mesures reçues", None),
            ("Alertes déclenchées", ui.fmt_int(len(alerts)), "collecte bloquée", None),
        ])
        fig = go.Figure(go.Bar(x=runs.run_ts, y=runs.failed, marker=dict(color=ui.BLUE),
                               hovertemplate="%{x|%d/%m à %H:%M} : %{y} anomalies<extra></extra>"))
        if not alerts.empty:
            fig.add_scatter(x=alerts.run_ts, y=alerts.failed, mode="markers", marker=dict(color="#d03b3b", size=12),
                            hovertemplate="%{x|%d/%m à %H:%M} : alerte déclenchée<extra></extra>")
        fig.update_yaxes(type="log", dtick=1, title_text="Anomalies détectées")
        fig.update_xaxes(tickformat="%d/%m")
        ui.plot(ui.style(fig, height=300, legend=False, hover="closest"), key="dq_runs")
        st.caption("Une barre par collecte (échelle logarithmique). Le point rouge marque une collecte où plus de 20 % "
                   "des données étaient anormales : elle a été bloquée et une alerte a été envoyée.")
        ui.insight("Les données anormales ne sont jamais supprimées en silence : elles sont mises de côté avec la "
                   "raison du rejet, pour pouvoir comprendre ce qui s'est passé. Une même donnée reçue deux fois n'est "
                   "comptée qu'une seule fois.")

    ui.section("Les outils utilisés")
    cols = st.columns(len(TOOLS))
    for col, (icon, name, desc) in zip(cols, TOOLS):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{icon} {name}**")
            st.caption(desc)
