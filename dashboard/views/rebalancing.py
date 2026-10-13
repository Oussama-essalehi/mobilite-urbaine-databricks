"""Rééquilibrage : plan d'action avant la prochaine pointe, calculé avec le code du projet (src/mobility)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data, live, ui
from views.realtime import LIVE, TABLE, current_state

DAYS = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]


def render() -> None:
    ui.page_header("Aide à la décision", "Rééquilibrage avant la prochaine pointe",
                   "Projection du stock de chaque station en fin de pointe (stock actuel + flux net attendu, issu de la "
                   "prévision ML si elle couvre la fenêtre, sinon du profil historique), seuils bas / haut "
                   "(2 vélos ou 15 % de la capacité), quantité pour revenir à 50 %, puis appariement glouton des "
                   "stations à délester vers les stations à réapprovisionner les plus proches (≤ 3 km).")
    profile = data.load("saturation_profile")
    if profile is None or profile.empty:
        ui.missing("Profils de saturation")
        return

    choice = st.segmented_control("État de départ", [TABLE, LIVE], default=TABLE, key="rb_source") or TABLE
    cur, stamp = current_state(choice)
    if cur is None:
        ui.missing("Disponibilité courante")
        return
    now = (pd.Timestamp(cur.snapshot_local.max()) if choice == LIVE and "API" in stamp else data.reference_time())
    plan = live.rebalancing_plan(cur, profile, data.load("forecast_station_hourly"), now)
    start, end, label, _ = plan["window"]
    state, moves = plan["state"], plan["moves"]
    if state.empty:
        st.info("Aucune station ouverte.")
        return

    src = state.flow_source.value_counts()
    st.markdown(f"#### :material/event: {label} · {DAYS[start.weekday()]} {start:%d/%m} {start:%H:%M}–{end:%H:%M}")
    st.caption(f"État de départ : {stamp} · flux attendu : " + ", ".join(f"{k} ({v} stations)" for k, v in src.items()))

    need = state[state.action == "REAPPROVISIONNER"]
    excess = state[state.action == "DELESTER"]
    covered = need.bikes_covered_by_pairing.sum() / need.bikes_to_move.sum() if need.bikes_to_move.sum() else 0
    dist = f"{ui.fmt_num(moves.distance_km.mean(), 1)} km en moy." if not moves.empty else None
    ui.kpis([
        ("À réapprovisionner", ui.fmt_int(len(need)), f"+{ui.fmt_int(need.bikes_to_move.sum())} vélos",
         "Stations sous le seuil bas en fin de pointe ; vélos à apporter pour revenir à 50 %", "off"),
        ("À délester", ui.fmt_int(len(excess)), f"-{ui.fmt_int(excess.bikes_to_move.sum())} vélos",
         "Stations au-dessus du seuil haut en fin de pointe ; vélos à retirer", "off"),
        ("Mouvements proposés", ui.fmt_int(len(moves)), dist, "Appariements station à délester → station à remplir", "off"),
        ("Besoin couvert", ui.fmt_pct(covered), None, "Part des vélos à apporter fournie par une station à délester ≤ 3 km"),
    ])
    if choice == LIVE and data.source().mode == "demo":
        st.caption(":material/info: État réel de l'API combiné aux profils et prévisions de la démo (simulés) : "
                   "le plan illustre la mécanique, pas un besoin opérationnel réel.")

    left, right = st.columns([1.35, 1])
    with left:
        shown = state.assign(action_label=state.action.map(ui.ACTION_LABEL))
        colors = {ui.ACTION_LABEL[k]: v for k, v in ui.ACTION.items()}
        shown = shown.assign(size=shown.bikes_to_move.clip(lower=1))
        fig = ui.station_map(shown.sort_values("action", ascending=False), "action_label", colors, size="size",
                             height=560, size_max=16,
                             hover={"available_bikes": True, "projected_bikes": ":.0f", "bikes_to_move": True,
                                    "flow_source": True, "size": False, "action_label": False, "lat": False, "lon": False})
        if not moves.empty:
            lat, lon = [], []
            for r in moves.itertuples():
                lat += [r.from_lat, r.to_lat, None]
                lon += [r.from_lon, r.to_lon, None]
            fig.add_trace(go.Scattermap(lat=lat, lon=lon, mode="lines", line=dict(width=1.5, color="rgba(82,81,78,0.75)"),
                                        name="Mouvement proposé", hoverinfo="skip"))
        ui.plot(fig, key="rb_map")
    with right:
        ui.section("Mouvements proposés", "Par station à réapprovisionner, de la plus prioritaire à la moins prioritaire")
        if moves.empty:
            st.info("Aucun appariement possible à moins de 3 km.")
        else:
            short = moves.assign(**{c: moves[c].str.replace(r"^\d+ - ", "", regex=True) for c in ["from_label", "to_label"]})
            st.dataframe(short[["from_label", "to_label", "bikes"]], hide_index=True, height=520,
                         width="stretch", column_config={
                             "from_label": "Depuis", "to_label": "Vers",
                             "bikes": st.column_config.NumberColumn("Vélos", width=50)})

    ui.section("Stations à traiter", "Triées par priorité = quantité × (1 + fréquence de saturation historique sur ce créneau)")
    todo = state[state.action != "OK"].sort_values("priority", ascending=False).assign(
        action=lambda d: d.action.map(ui.ACTION_LABEL))
    st.dataframe(todo[["station_label", "action", "available_bikes", "expected_net_flow", "projected_bikes", "capacity",
                       "bikes_to_move", "bikes_covered_by_pairing", "priority"]], hide_index=True, width="stretch",
                 height=360, column_config={
                     "station_label": "Station", "action": "Action", "available_bikes": "Vélos actuels",
                     "expected_net_flow": st.column_config.NumberColumn("Flux net attendu", format="%+.1f"),
                     "projected_bikes": st.column_config.NumberColumn("Vélos projetés", format="%.0f"),
                     "capacity": "Capacité", "bikes_to_move": "À déplacer", "bikes_covered_by_pairing": "Couverts",
                     "priority": st.column_config.NumberColumn("Priorité", format="%.1f")})
    st.caption("Logique : `src/mobility/rebalancing.py` (fonctions testées unitairement : `next_peak_window`, "
               "`classify_station`, `pair_moves`) — identique au notebook `07_recommendations.py`.")
