"""Les vélos en ce moment : carte des stations et leur état (en direct ou dernière collecte enregistrée)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data, live, ui

LIVE, TABLE = "En direct", "Dernière collecte enregistrée"


def current_state(choice: str) -> tuple[pd.DataFrame | None, str]:
    """État des stations selon la source choisie, avec le nom et la famille de chaque station."""
    dim = data.load("dim_station")
    if choice == LIVE:
        res = live.fetch_live_status()
        if res is not None:
            df, stamp = res
            if dim is not None:
                df = df.merge(dim[["station_id", "station_label", "cluster_label"]], on="station_id", how="left")
            else:
                df["station_label"], df["cluster_label"] = None, None
            df["station_label"] = df.station_label.fillna("Station " + df.station_id.astype(str))
            return df, stamp
        st.warning("Le service VélôToulouse ne répond pas : affichage de la dernière collecte enregistrée.",
                   icon=":material/cloud_off:")
    cur = data.load("station_current")
    if cur is None:
        return None, ""
    return cur, f"{pd.Timestamp(cur.snapshot_local.max()):%d/%m/%Y à %H:%M}"


def render() -> None:
    ui.page_header("Comprendre", "Les vélos en ce moment",
                   "Chaque point est une station. Sa couleur indique s'il est facile d'y prendre un vélo ou d'en "
                   "rendre un ; sa taille correspond au nombre de places de la station.")

    choice = st.segmented_control("Données affichées", [LIVE, TABLE], default=LIVE, key="rt_source") or LIVE
    df, stamp = current_state(choice)
    if df is None:
        ui.missing("L'état des stations")
        return

    open_ = df[df.status == "OPEN"]
    counts = df.availability_status.value_counts()
    ui.kpis([
        ("Vélos disponibles", ui.fmt_int(open_.available_bikes.sum()), "dans toute la ville", None),
        ("Places libres", ui.fmt_int(open_.available_stands.sum()), "pour rendre un vélo", None),
        ("Stations vides", ui.fmt_int(counts.get("Vide", 0)), "aucun vélo à prendre", None),
        ("Stations pleines", ui.fmt_int(counts.get("Pleine", 0)), "aucune place pour rendre", None),
    ])
    st.caption(f":material/schedule: Situation au {stamp}")

    left, right = st.columns([2.3, 1])
    with left:
        fig = ui.station_map(df, "availability_status", ui.AVAILABILITY, size="usable_capacity", height=560,
                             hover={"available_bikes": True, "available_stands": True, "usable_capacity": False,
                                    "availability_status": False, "lat": False, "lon": False},
                             labels={"available_bikes": "Vélos", "available_stands": "Places libres"})
        ui.plot(fig, key="rt_map")
    with right:
        ui.section("Nombre de stations par état")
        order = [s for s in ui.AVAILABILITY if s in counts.index]
        bar = go.Figure(go.Bar(
            y=order, x=[counts[s] for s in order], orientation="h",
            marker=dict(color=[ui.AVAILABILITY[s] for s in order], cornerradius=4),
            text=[ui.fmt_int(counts[s]) for s in order], textposition="outside", cliponaxis=False,
            hovertemplate="%{y} : %{x} stations<extra></extra>"))
        bar.update_yaxes(autorange="reversed", showgrid=False)
        bar.update_xaxes(showgrid=False, showticklabels=False, range=[0, counts.max() * 1.2])
        ui.plot(ui.style(bar, height=250, legend=False, hover="closest"), key="rt_bar")
        st.caption("« Presque vide » : 2 vélos ou moins. « Presque pleine » : 2 places libres ou moins.")

        ui.section("Stations vides ou pleines")
        watch = (open_[open_.availability_status.isin(["Vide", "Pleine"])]
                 .sort_values(["availability_status", "usable_capacity"], ascending=[False, False]))
        ui.table(watch, {"station_label": "Station", "availability_status": "État"}, height=230)

    critical = open_.availability_status.isin(["Vide", "Pleine"]).mean()
    near = open_.availability_status.isin(["Presque vide", "Presque pleine"]).mean()
    ui.insight(f"En ce moment, <b>{ui.fmt_pct(critical)}</b> des stations sont vides ou pleines, et "
               f"<b>{ui.fmt_pct(near)}</b> sont sur le point de l'être. C'est ce déséquilibre que la page "
               "<i>Déplacer les vélos</i> cherche à corriger.")
