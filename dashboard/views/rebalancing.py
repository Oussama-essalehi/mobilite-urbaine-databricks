"""Déplacer les vélos : plan conseillé avant la prochaine heure de pointe (code du projet, src/mobility)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data, live, ui
from views.realtime import LIVE, TABLE, current_state

DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
WINDOWS = {"Pointe du matin": "l'heure de pointe du matin", "Pointe du soir": "l'heure de pointe du soir",
           "Après-midi week-end": "l'après-midi du week-end"}


def short(s: pd.Series) -> pd.Series:
    return s.str.replace(r"^\d+ - ", "", regex=True)


def render() -> None:
    ui.page_header("Agir", "Déplacer les vélos avant l'heure de pointe",
                   "Avant chaque heure de pointe, l'application estime combien de vélos il restera dans chaque station "
                   "à la fin de la pointe. Elle repère celles qui risquent d'être vides ou pleines, puis propose de "
                   "déplacer des vélos d'une station trop pleine vers une station proche (moins de 3 km) qui en manquera.")
    profile = data.load("saturation_profile")
    if profile is None or profile.empty:
        ui.missing("Les habitudes des stations")
        return

    choice = st.segmented_control("Point de départ", [TABLE, LIVE], default=TABLE, key="rb_source") or TABLE
    cur, stamp = current_state(choice)
    if cur is None:
        ui.missing("L'état des stations")
        return
    now = pd.Timestamp(cur.snapshot_local.max()) if choice == LIVE else data.reference_time()
    plan = live.rebalancing_plan(cur, profile, data.load("forecast_station_hourly"), now)
    start, end, label, _ = plan["window"]
    state, moves = plan["state"], plan["moves"]
    if state.empty:
        st.info("Aucune station ouverte pour le moment.")
        return

    st.markdown(f"#### :material/event: Préparer {WINDOWS.get(label, label)} : {DAYS[start.weekday()]} "
                f"{start:%d/%m}, de {start:%H} h à {end:%H} h")
    st.caption(f"État des stations au {stamp}.")
    if choice == LIVE and data.source().mode == "demo":
        st.caption(":material/info: L'état réel des stations est combiné ici aux habitudes simulées de la "
                   "démonstration : le plan illustre la méthode, ce n'est pas une consigne réelle.")

    need = state[state.action == "REAPPROVISIONNER"]
    excess = state[state.action == "DELESTER"]
    covered = need.bikes_covered_by_pairing.sum() / need.bikes_to_move.sum() if need.bikes_to_move.sum() else 0
    ui.kpis([
        ("Stations à remplir", ui.fmt_int(len(need)), f"{ui.fmt_int(need.bikes_to_move.sum())} vélos à apporter", None),
        ("Stations à alléger", ui.fmt_int(len(excess)), f"{ui.fmt_int(excess.bikes_to_move.sum())} vélos à retirer", None),
        ("Trajets de camion", ui.fmt_int(len(moves)),
         f"{ui.fmt_num(moves.distance_km.mean(), 1)} km en moyenne" if not moves.empty else None, None),
        ("Besoin couvert", ui.fmt_pct(covered), "grâce aux stations proches",
         "Part des vélos à apporter qui peuvent venir d'une station trop pleine à moins de 3 km"),
    ])

    left, right = st.columns([1.35, 1])
    with left:
        colors = {ui.ACTION_LABEL[k]: v for k, v in ui.ACTION.items()}
        shown = state.assign(action_label=state.action.map(ui.ACTION_LABEL), size=state.bikes_to_move.clip(lower=1))
        fig = ui.station_map(shown.sort_values("action", ascending=False), "action_label", colors, size="size",
                             height=560, size_max=16,
                             hover={"available_bikes": True, "projected_bikes": ":.0f", "bikes_to_move": True,
                                    "size": False, "action_label": False, "lat": False, "lon": False},
                             labels={"available_bikes": "Vélos maintenant", "projected_bikes": "Vélos en fin de pointe",
                                     "bikes_to_move": "Vélos à déplacer"})
        if not moves.empty:
            lat, lon = [], []
            for r in moves.itertuples():
                lat += [r.from_lat, r.to_lat, None]
                lon += [r.from_lon, r.to_lon, None]
            fig.add_trace(go.Scattermap(lat=lat, lon=lon, mode="lines", line=dict(width=2, color=ui.MAP_LINE),
                                        name="Trajet de camion", hoverinfo="skip"))
        ui.plot(fig, key="rb_map")
    with right:
        ui.section("Trajets conseillés", "Les stations les plus urgentes d'abord")
        if moves.empty:
            st.info("Aucun déplacement possible à moins de 3 km.")
        else:
            m = moves.assign(frm=short(moves.from_label), to=short(moves.to_label))
            ui.table(m, {"frm": "Prendre à", "to": "Déposer à", "bikes": "Vélos"}, num=("bikes",), height=500)

    ui.insight(f"En déplaçant <b>{ui.fmt_int(moves.bikes.sum() if not moves.empty else 0)} vélos</b> avant la pointe, "
               f"on couvre <b>{ui.fmt_pct(covered)}</b> des besoins des stations qui en manqueront. Le reste ne peut "
               "pas être fourni par une station voisine trop pleine.")
