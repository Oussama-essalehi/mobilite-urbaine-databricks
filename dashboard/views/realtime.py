"""Disponibilité en direct : carte des stations et statuts (gold_station_current ou flux GBFS live)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data, live, ui

LIVE, TABLE = "Flux GBFS en direct", "Table gold_station_current"


def current_state(choice: str) -> tuple[pd.DataFrame | None, str]:
    """État des stations selon la source choisie, enrichi du libellé et du segment."""
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
            return df, f"API VélôToulouse · {stamp}"
        st.warning("Le flux GBFS ne répond pas : affichage de la dernière table gold.", icon=":material/cloud_off:")
    cur = data.load("station_current")
    if cur is None:
        return None, ""
    return cur, f"gold_station_current · {pd.Timestamp(cur.snapshot_local.max()):%d/%m/%Y %H:%M}"


def render() -> None:
    ui.page_header("Analytique · temps réel", "Disponibilité en direct",
                   "État de chaque station : vélos disponibles, bornes libres et statut. Les statuts reprennent les "
                   "règles de la table <code>gold_station_current</code> (vide, presque vide ≤ 2 vélos, pleine, "
                   "presque pleine ≤ 2 bornes).")

    c1, c2 = st.columns([1.3, 2])
    choice = c1.segmented_control("Source", [LIVE, TABLE], default=LIVE, key="rt_source")
    statuses = list(ui.AVAILABILITY)
    selected = c2.pills("Statuts affichés", statuses, default=statuses, selection_mode="multi", key="rt_status")

    df, stamp = current_state(choice or LIVE)
    if df is None:
        ui.missing("Disponibilité courante")
        return

    open_ = df[df.status == "OPEN"]
    counts = df.availability_status.value_counts()
    ui.kpis([
        ("Vélos disponibles", ui.fmt_int(open_.available_bikes.sum()), None, "Somme sur les stations ouvertes"),
        ("Bornes libres", ui.fmt_int(open_.available_stands.sum()), None, None),
        ("Stations vides", ui.fmt_int(counts.get("Vide", 0)), None, "Aucun vélo disponible"),
        ("Stations pleines", ui.fmt_int(counts.get("Pleine", 0)), None, "Aucune borne libre pour déposer"),
        ("Remplissage", ui.fmt_pct(open_.fill_rate.mean()), None, "vélos / (vélos + bornes libres)"),
    ])
    st.caption(f":material/schedule: Dernière collecte : {stamp}")

    critical = open_.availability_status.isin(["Vide", "Pleine"]).mean()
    near = open_.availability_status.isin(["Presque vide", "Presque pleine"]).mean()
    ui.insight(f"<b>{ui.fmt_pct(critical)}</b> des stations ouvertes sont vides ou pleines en ce moment, et "
               f"<b>{ui.fmt_pct(near)}</b> sont à 2 vélos ou 2 bornes de l'être.")

    shown = df[df.availability_status.isin(selected or [])]
    left, right = st.columns([2.3, 1])
    with left:
        fig = ui.station_map(shown, "availability_status", ui.AVAILABILITY, size="usable_capacity", height=560,
                             hover={"available_bikes": True, "available_stands": True, "usable_capacity": False,
                                    "availability_status": False, "lat": False, "lon": False})
        ui.plot(fig, key="rt_map")
    with right:
        order = [s for s in ui.AVAILABILITY if s in counts.index]
        bar = go.Figure(go.Bar(
            y=order, x=[counts[s] for s in order], orientation="h",
            marker=dict(color=[ui.AVAILABILITY[s] for s in order], cornerradius=4),
            text=[ui.fmt_int(counts[s]) for s in order], textposition="outside", cliponaxis=False,
            hovertemplate="%{y} : %{x} stations<extra></extra>"))
        bar.update_yaxes(autorange="reversed", showgrid=False)
        bar.update_xaxes(showgrid=False, showticklabels=False, range=[0, counts.max() * 1.2])
        ui.plot(ui.style(bar, height=260, legend=False, hover="closest"), key="rt_bar")

        ui.section("Stations à surveiller", "Vides ou pleines, par capacité décroissante")
        watch = (open_[open_.availability_status.isin(["Vide", "Pleine"])]
                 .sort_values("usable_capacity", ascending=False)
                 [["station_label", "availability_status"]])
        st.dataframe(watch, hide_index=True, height=230, width="stretch",
                     column_config={"station_label": "Station", "availability_status": "Statut"})

    ui.data_table(df.sort_values("station_id")[[
        "station_label", "cluster_label", "availability_status", "available_bikes", "available_stands",
        "usable_capacity", "fill_rate"]], column_config={
        "fill_rate": st.column_config.ProgressColumn("Remplissage", format="percent", min_value=0, max_value=1)})
