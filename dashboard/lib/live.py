"""Disponibilité en direct depuis le flux GBFS public VélôToulouse, et rééquilibrage calculé à la volée.

Le parsing reprend les règles de `parse_velo` (03_silver.py) et le statut de disponibilité celles de
`gold_station_current` (04_gold.py). Le rééquilibrage appelle les fonctions du projet (`src/mobility`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mobility.rebalancing import classify_station, next_peak_window, pair_moves  # noqa: E402

GBFS = "https://api.cyclocity.fr/contracts/toulouse/gbfs/v3/{feed}.json"
MAX_PAIR_KM = 3.0


def availability_status(df: pd.DataFrame) -> np.ndarray:
    return np.select(
        [df.status != "OPEN", df.available_bikes == 0, df.available_stands == 0, df.available_bikes <= 2,
         df.available_stands <= 2],
        ["Fermée", "Vide", "Pleine", "Presque vide", "Presque pleine"], "Disponible")


@st.cache_data(ttl=60, show_spinner="Lecture de l'état des stations VélôToulouse…")
def fetch_live_status() -> tuple[pd.DataFrame, str] | None:
    """État courant de toutes les stations, ou None si l'API ne répond pas."""
    try:
        status = requests.get(GBFS.format(feed="station_status"), timeout=10).json()
        info = requests.get(GBFS.format(feed="station_information"), timeout=10).json()
    except Exception:  # noqa: BLE001
        return None
    s = pd.DataFrame(status["data"]["stations"])
    i = pd.DataFrame(info["data"]["stations"])
    if s.empty or i.empty:
        return None
    s["station_id"] = s.station_id.astype(str).str.extract(r"(\d+)$")[0].astype(int)
    i["station_id"] = i.station_id.astype(str).str.extract(r"(\d+)$")[0].astype(int)
    bikes_col = "num_vehicles_available" if "num_vehicles_available" in s else "num_bikes_available"
    df = s.merge(i[["station_id", "lat", "lon", "capacity"]], on="station_id", how="inner")
    df["available_bikes"] = df[bikes_col].astype(int)
    df["available_stands"] = df.num_docks_available.astype(int)
    df["status"] = np.where(df.is_installed.astype(bool) & df.is_renting.astype(bool), "OPEN", "CLOSED")
    df["usable_capacity"] = df.available_bikes + df.available_stands
    df["fill_rate"] = df.available_bikes / df.usable_capacity.replace(0, np.nan)
    df["availability_status"] = availability_status(df)
    stamp = pd.Timestamp(status.get("last_updated")).tz_convert("Europe/Paris").tz_localize(None)
    df["snapshot_local"] = stamp
    cols = ["station_id", "lat", "lon", "snapshot_local", "status", "available_bikes", "available_stands",
            "usable_capacity", "fill_rate", "availability_status"]
    return df[cols], f"{stamp:%d/%m/%Y à %H:%M}"


def rebalancing_plan(current: pd.DataFrame, profile: pd.DataFrame, forecast: pd.DataFrame | None,
                     now_local: pd.Timestamp) -> dict:
    """Même logique que 07_recommendations.py, en pandas.

    current  : état des stations (gold_station_current ou flux live enrichi des libellés)
    profile  : gold_saturation_profile
    forecast : gold_forecast_station_hourly (facultatif)
    """
    start, end, label = next_peak_window(now_local.to_pydatetime())
    day_type = "Week-end" if start.weekday() >= 5 else "Semaine"
    h0 = now_local.floor("h") + pd.Timedelta(hours=1)
    end_ts = pd.Timestamp(end)
    hours = pd.date_range(h0, end_ts - pd.Timedelta(hours=1), freq="h")
    n_expected = len(hours)

    # flux net attendu d'ici la fin de la pointe : prévision ML si elle couvre la fenêtre, sinon profil historique
    fc = pd.DataFrame(columns=["station_id", "fc_net", "n_hours"])
    if forecast is not None and not forecast.empty:
        f = forecast[(forecast.hour_local >= h0) & (forecast.hour_local < end_ts)]
        fc = f.groupby("station_id").agg(fc_net=("pred_net_flow", "sum"), n_hours=("pred_net_flow", "size")).reset_index()

    keys = pd.DataFrame({"hour_of_day": hours.hour,
                         "day_type": np.where(hours.dayofweek >= 5, "Week-end", "Semaine")})
    hist = (keys.merge(profile, on=["hour_of_day", "day_type"]).groupby("station_id").avg_net_flow.sum()
            .rename("hist_net").reset_index())
    win = profile[(profile.day_type == day_type) & (profile.hour_of_day >= start.hour) & (profile.hour_of_day < end.hour)]
    sat = win.groupby("station_id").agg(hist_pct_empty=("pct_time_empty", "mean"),
                                        hist_pct_full=("pct_time_full", "mean")).reset_index()

    state = (current[(current.status == "OPEN") & (current.usable_capacity > 0)]
             .merge(fc, on="station_id", how="left").merge(hist, on="station_id", how="left")
             .merge(sat, on="station_id", how="left"))
    use_fc = state.n_hours.fillna(0) >= max(n_expected, 1)
    state["expected_net_flow"] = np.where(use_fc, state.fc_net, state.hist_net)
    state["flow_source"] = np.where(use_fc, "Prévision ML", "Profil historique")
    state["expected_net_flow"] = state.expected_net_flow.astype(float).fillna(0.0)
    state[["hist_pct_empty", "hist_pct_full"]] = state[["hist_pct_empty", "hist_pct_full"]].fillna(0)
    state["capacity"] = state.usable_capacity
    state["projected_bikes"] = np.minimum((state.available_bikes + state.expected_net_flow).clip(lower=0), state.capacity)

    if state.empty:
        return {"window": (start, end, label, day_type), "state": state, "moves": pd.DataFrame()}

    actions = state.apply(lambda r: classify_station(r.projected_bikes, int(r.capacity)), axis=1, result_type="expand")
    state[["action", "bikes_to_move"]] = actions
    state["priority"] = np.where(state.action == "REAPPROVISIONNER", state.bikes_to_move * (1 + state.hist_pct_empty),
                                 state.bikes_to_move * (1 + state.hist_pct_full))

    def nodes(df):
        return [{"station_id": int(r.station_id), "lat": r.lat, "lon": r.lon, "qty": int(r.bikes_to_move),
                 "priority": float(r.priority)} for r in df.itertuples()]

    moves = pd.DataFrame(pair_moves(receivers=nodes(state[state.action == "REAPPROVISIONNER"]),
                                    donors=nodes(state[state.action == "DELESTER"]), max_km=MAX_PAIR_KM),
                         columns=["from_station", "to_station", "bikes", "distance_km"])
    served = moves.groupby("to_station").bikes.sum() if not moves.empty else pd.Series(dtype=float)
    state["bikes_covered_by_pairing"] = np.where(state.action == "REAPPROVISIONNER",
                                                 state.station_id.map(served).fillna(0), 0).astype(int)
    if not moves.empty:
        pos = state.set_index("station_id")
        for side, col in (("from", "from_station"), ("to", "to_station")):
            moves[f"{side}_label"] = moves[col].map(pos.station_label)
            moves[f"{side}_lat"] = moves[col].map(pos.lat)
            moves[f"{side}_lon"] = moves[col].map(pos.lon)
    return {"window": (start, end, label, day_type), "state": state, "moves": moves}
