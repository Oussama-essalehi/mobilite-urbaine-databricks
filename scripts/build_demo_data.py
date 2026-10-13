"""Construit l'instantané de démonstration du tableau de bord Streamlit (`dashboard/demo_data/`).

Le tableau de bord lit normalement les tables gold du SQL warehouse Databricks. Pour qu'il soit consultable
sans workspace, ce script produit des tables **au même schéma** :

- les stations (nom, position, capacité) viennent du flux GBFS public VélôToulouse ;
- l'historique horaire (remplissage, départs, arrivées, météo) est **simulé** : profils de stations
  résidentielles / pôles d'activité / équilibrées / souvent vides / souvent pleines, courbes de demande
  semaine et week-end, effet de la pluie et de la température ;
- les tables dérivées sont recalculées comme dans `src/04_gold.py`, le clustering et la prévision avec les
  mêmes modèles et les mêmes fonctions que `05_ml_clustering.py` et `06_ml_forecast.py`.

Usage :  python scripts/build_demo_data.py [--offline]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mobility.ml_utils import (  # noqa: E402
    add_station_encodings,
    dedupe_labels,
    name_cluster,
    regression_metrics,
)

OUT = ROOT / "dashboard" / "demo_data"
GBFS = "https://api.cyclocity.fr/contracts/toulouse/gbfs/v3/station_information.json"
PARIS = ZoneInfo("Europe/Paris")
CENTRE = (43.6045, 1.4440)  # Capitole
HISTORY_DAYS = 42
SEED = 42
rng = np.random.default_rng(SEED)

PEAK_HOURS = [7, 8, 9, 17, 18, 19]

# mêmes paramètres que src/06_ml_forecast.py
TARGETS = ["departures", "arrivals"]
FEATURES = ["hour_of_day", "weekday", "is_weekend", "temperature", "precipitation", "wind_speed",
            "lag_24h", "lag_168h", "station_mean", "station_hour_mean", "capacity", "cluster_id"]
PARAMS = {"loss": "poisson", "learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31,
          "min_samples_leaf": 40, "l2_regularization": 1.0, "early_stopping": True,
          "validation_fraction": 0.1, "random_state": 42}


# ---------------------------------------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------------------------------------

def haversine_km(lat, lon, lat0, lon0):
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def load_stations(offline: bool) -> pd.DataFrame:
    if not offline:
        try:
            raw = requests.get(GBFS, timeout=30).json()["data"]["stations"]
            df = pd.DataFrame({
                "station_id": [int(s["station_id"].split(":")[-1]) for s in raw],
                "station_name": [
                    (s["name"][0]["text"] if isinstance(s["name"], list) else s["name"]) for s in raw],
                "address": [s.get("address") for s in raw],
                "lat": [float(s["lat"]) for s in raw],
                "lon": [float(s["lon"]) for s in raw],
                "capacity": [int(s.get("capacity") or 20) for s in raw],
                "banking": ["creditcard" in (s.get("rental_methods") or []) for s in raw],
            })
            # même nettoyage que parse_velo (03_silver.py)
            df["station_name"] = (df["station_name"].str.replace(r"^[0-9]+\s*-\s*", "", regex=True)
                                  .str.strip().str.title())
            df["address"] = df["address"].fillna("").str.strip().str.title()
            df["station_name"] = df.station_name.where(df.station_name.str.len() > 0,
                                                       df.address.where(df.address.str.len() > 0, "Sans nom"))
            df = df[df.lat.between(43.45, 43.75) & df.lon.between(1.25, 1.65) & (df.capacity > 0)]
            print(f"{len(df)} stations lues depuis le flux GBFS")
            return df.sort_values("station_id").reset_index(drop=True)
        except Exception as e:  # noqa: BLE001
            print(f"GBFS indisponible ({e}) — stations synthétiques")
    n = 280
    r = rng.gamma(2.0, 1.1, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    return pd.DataFrame({
        "station_id": np.arange(1, n + 1),
        "station_name": [f"Station {i:03d}" for i in range(1, n + 1)],
        "address": "",
        "lat": CENTRE[0] + r * np.sin(theta) / 111,
        "lon": CENTRE[1] + r * np.cos(theta) / 80,
        "capacity": rng.choice([15, 18, 20, 22, 25, 30], n),
        "banking": rng.random(n) < 0.4,
    })


# ---------------------------------------------------------------------------------------------------------
# Profils horaires (points de passage interpolés sur 24 h)
# ---------------------------------------------------------------------------------------------------------

def curve(points):
    h, v = zip(*points)
    return np.interp(np.arange(24), h, v)


FILL = {  # remplissage attendu (semaine, week-end) par archétype
    "residential": (curve([(0, .68), (6, .70), (7, .60), (8, .40), (9, .28), (12, .30), (16, .33), (17, .40),
                           (18, .55), (19, .63), (21, .68), (23, .68)]),
                    curve([(0, .66), (9, .64), (11, .53), (14, .43), (17, .45), (19, .55), (22, .63), (23, .66)])),
    "business":    (curve([(0, .25), (6, .24), (7, .32), (8, .58), (9, .78), (12, .74), (16, .72), (17, .60),
                           (18, .38), (19, .28), (22, .25), (23, .25)]),
                    curve([(0, .32), (10, .33), (14, .38), (18, .35), (23, .32)])),
    "balanced":    (curve([(0, .50), (8, .46), (12, .53), (15, .55), (19, .45), (22, .50), (23, .50)]),
                    curve([(0, .48), (11, .50), (14, .60), (17, .66), (20, .58), (23, .50)])),
    "empty":       (curve([(0, .14), (7, .10), (9, .05), (13, .07), (18, .09), (23, .14)]),
                    curve([(0, .12), (12, .08), (18, .07), (23, .12)])),
    "full":        (curve([(0, .90), (8, .88), (10, .94), (16, .93), (19, .88), (23, .90)]),
                    curve([(0, .88), (14, .92), (18, .95), (23, .88)])),
}

DEMAND = (  # intensité relative des départs (semaine, week-end)
    curve([(0, .15), (5, .05), (6, .30), (7, 1.0), (8, 1.9), (9, 1.2), (10, .70), (12, 1.0), (13, .95),
           (14, .75), (16, .95), (17, 1.6), (18, 2.0), (19, 1.3), (20, .80), (22, .50), (23, .25)]),
    curve([(0, .40), (3, .15), (7, .10), (9, .35), (11, .80), (14, 1.2), (16, 1.35), (18, 1.2), (20, .85),
           (23, .50)]),
)


def assign_archetypes(stations: pd.DataFrame) -> np.ndarray:
    d = haversine_km(stations.lat.to_numpy(), stations.lon.to_numpy(), *CENTRE)
    kinds = np.array(["balanced", "business", "full", "empty", "residential"])
    probs = np.where(
        (d < 1.5)[:, None], [.42, .25, .15, .06, .12],
        np.where((d < 4)[:, None], [.20, .20, .05, .10, .45], [.10, .20, .02, .20, .48]))
    return np.array([rng.choice(kinds, p=p) for p in probs]), d


# ---------------------------------------------------------------------------------------------------------
# Météo horaire
# ---------------------------------------------------------------------------------------------------------

def simulate_weather(hours: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(hours)
    days = hours.normalize()
    uniq = days.unique()
    # anomalie journalière AR(1) et tendance saisonnière (fin d'été → automne toulousain)
    anom = np.zeros(len(uniq))
    for i in range(1, len(uniq)):
        anom[i] = 0.7 * anom[i - 1] + rng.normal(0, 1.8)
    trend = np.linspace(21.5, 15.5, len(uniq))
    rainy_day = rng.random(len(uniq)) < 0.30
    day_idx = pd.Index(uniq).get_indexer(days)

    hod = hours.hour.to_numpy()
    temp = trend[day_idx] + anom[day_idx] + 5.5 * np.sin(2 * np.pi * (hod - 10) / 24) + rng.normal(0, .5, n)

    raining = np.zeros(n, bool)
    state = False
    for i in range(n):
        p_start = 0.12 if rainy_day[day_idx[i]] else 0.004
        state = (rng.random() < 0.72) if state else (rng.random() < p_start)
        raining[i] = state
    precip = np.where(raining, rng.gamma(1.3, 1.0, n), 0.0).round(1)
    temp = temp - 2.0 * raining

    cloud_code = rng.choice([0, 1, 2, 3], n, p=[.30, .25, .25, .20])
    code = np.where(precip >= 2, 63, np.where(precip > 0.1, 61, cloud_code))
    df = pd.DataFrame({
        "hour_local": hours,
        "temperature": temp.round(1),
        "precipitation": precip,
        "wind_speed": (6 + rng.gamma(2.0, 4.0, n)).round(1),
        "weather_code": code,
    })
    # même règle que parse_meteo (03_silver.py)
    df["weather_category"] = np.select(
        [df.precipitation >= 2, df.precipitation > 0.1, df.weather_code.isin([0, 1])],
        ["Forte pluie", "Pluie", "Dégagé"], "Nuageux")
    return df


def demand_factor(temp, precip):
    rain = np.where(precip >= 2, 0.45, np.where(precip > 0.1, 0.68, 1.0))
    t = np.interp(temp, [0, 5, 10, 15, 22, 28, 34], [.62, .76, .89, 1.0, 1.07, 1.02, .88])
    return rain * t


# ---------------------------------------------------------------------------------------------------------
# Table de faits horaire (équivalent de gold_station_hourly)
# ---------------------------------------------------------------------------------------------------------

def simulate_station_hourly(stations, kinds, dist, hours, weather):
    S, H = len(stations), len(hours)
    cap = stations.capacity.to_numpy()[:, None].astype(float)
    hod = hours.hour.to_numpy()
    is_we = np.asarray(hours.dayofweek >= 5)
    day_idx = pd.Index(hours.normalize().unique()).get_indexer(hours.normalize())
    n_days = day_idx.max() + 1

    base_fill = np.zeros((S, H))
    next_fill = np.zeros((S, H))
    for k, (wd, we) in FILL.items():
        m = kinds == k
        prof = np.where(is_we, we[hod], wd[hod])
        prof_next = np.where(is_we, we[(hod + 1) % 24], wd[(hod + 1) % 24])
        base_fill[m] = prof
        next_fill[m] = prof_next

    st_offset = rng.normal(0, .05, (S, 1))
    day_noise = rng.normal(0, .045, (S, n_days))[:, day_idx]
    fill = np.clip(base_fill + st_offset + day_noise + rng.normal(0, .03, (S, H)), 0, 1)

    # probabilité d'être vide / pleine : loi binomiale négative (r = 2,5) sur le stock moyen
    lam_b, lam_d = fill * cap, (1 - fill) * cap
    p_empty = (1 + lam_b / 2.5) ** -2.5
    p_full = (1 + lam_d / 2.5) ** -2.5
    n_snap = np.where(rng.random((S, H)) < 0.03, 3, 4)
    pct_empty = rng.binomial(n_snap, p_empty) / n_snap
    pct_full = rng.binomial(n_snap, p_full) / n_snap

    # activité de la station : centrale et grande = plus de flux
    activity = (np.exp(rng.normal(0, .45, S)) * (cap[:, 0] / 20) * np.interp(dist, [0, 2, 5, 9], [1.7, 1.2, .8, .55]))
    intensity = np.where(is_we, DEMAND[1][hod] * 0.85, DEMAND[0][hod])
    wf = demand_factor(weather.temperature.to_numpy(), weather.precipitation.to_numpy())
    base = activity[:, None] * intensity[None, :] * wf[None, :] * 0.95
    net = (next_fill - base_fill) * cap  # flux net induit par le profil
    departures = rng.poisson(base + np.clip(-net, 0, None) * wf)
    arrivals = rng.poisson(base + np.clip(net, 0, None) * wf)

    # pannes : quelques stations fermées une journée
    closed_day = rng.random((S, n_days)) < 0.008
    closed = closed_day[:, day_idx]
    pct_open = np.where(closed, 0.0, 1.0)
    for arr in (departures, arrivals, pct_empty, pct_full):
        arr[closed] = 0

    df = pd.DataFrame({
        "station_id": np.repeat(stations.station_id.to_numpy(), H),
        "hour_local": np.tile(hours.to_numpy(), S),
        "n_snapshots": n_snap.ravel(),
        "pct_open": pct_open.ravel(),
        "avg_bikes": (fill * cap).ravel().round(2),
        "avg_fill_rate": fill.ravel(),
        "pct_time_empty": pct_empty.ravel(),
        "pct_time_full": pct_full.ravel(),
        "departures": departures.ravel(),
        "arrivals": arrivals.ravel(),
    })
    df["net_flow"] = df.arrivals - df.departures
    df["date_local"] = df.hour_local.dt.normalize()
    df["hour_of_day"] = df.hour_local.dt.hour
    df["weekday"] = df.hour_local.dt.dayofweek + 1
    df["is_weekend"] = (df.weekday >= 6).astype(int)
    df["day_type"] = np.where(df.is_weekend == 1, "Week-end", "Semaine")
    return df.merge(weather, on="hour_local", how="left")


# ---------------------------------------------------------------------------------------------------------
# Tables gold dérivées (mêmes définitions que src/04_gold.py)
# ---------------------------------------------------------------------------------------------------------

def network_hourly(sh):
    g = sh.groupby(["hour_local", "date_local", "hour_of_day", "weekday", "day_type", "is_weekend"])
    out = g.agg(total_departures=("departures", "sum"), total_arrivals=("arrivals", "sum"),
                avg_fill_rate=("avg_fill_rate", "mean"),
                stations_mostly_empty=("pct_time_empty", lambda s: int((s >= .5).sum())),
                stations_mostly_full=("pct_time_full", lambda s: int((s >= .5).sum())),
                stations_reporting=("station_id", "size"),
                temperature=("temperature", "first"), precipitation=("precipitation", "first"),
                wind_speed=("wind_speed", "first"), weather_category=("weather_category", "first")).reset_index()
    out["temperature_band"] = pd.cut(out.temperature, [-99, 5, 15, 25, 99], right=False,
                                     labels=["1. < 5 °C", "2. 5-15 °C", "3. 15-25 °C", "4. > 25 °C"]).astype(str)
    return out


def saturation_profile(sh, today):
    w = sh[(sh.date_local >= today - pd.Timedelta(days=28)) & (sh.pct_open > .5)]
    out = (w.groupby(["station_id", "day_type", "hour_of_day"])
           .agg(n_hours=("departures", "size"), avg_fill_rate=("avg_fill_rate", "mean"),
                pct_time_empty=("pct_time_empty", "mean"), pct_time_full=("pct_time_full", "mean"),
                avg_departures=("departures", "mean"), avg_arrivals=("arrivals", "mean"),
                avg_net_flow=("net_flow", "mean")).reset_index())
    out["saturation_status"] = np.select(
        [out.pct_time_empty >= .25, out.pct_time_full >= .25], ["Souvent vide", "Souvent pleine"], "Normale")
    return out


def saturation_ranking(sh, dim, today):
    w = sh[(sh.date_local >= today - pd.Timedelta(days=28)) & (sh.day_type == "Semaine")
           & sh.hour_of_day.isin(PEAK_HOURS) & (sh.pct_open > .5)]
    p = w.groupby("station_id").agg(pct_empty_peak=("pct_time_empty", "mean"),
                                    pct_full_peak=("pct_time_full", "mean")).reset_index()
    p["avg_net_flow_morning"] = p.station_id.map(w[w.hour_of_day < 12].groupby("station_id").net_flow.mean())
    p["avg_net_flow_evening"] = p.station_id.map(w[w.hour_of_day >= 12].groupby("station_id").net_flow.mean())
    p = p.merge(dim[["station_id", "station_label", "lat", "lon", "cluster_label"]], on="station_id")
    p["pct_saturated_peak"] = p.pct_empty_peak + p.pct_full_peak
    p["saturation_rank"] = p.pct_saturated_peak.rank(method="dense", ascending=False).astype(int)
    p["main_issue"] = np.where(p.pct_empty_peak >= p.pct_full_peak, "Manque de vélos", "Manque de bornes")
    return p.sort_values("saturation_rank").reset_index(drop=True)


def weather_effect(net):
    base = net[net.weather_category.notna() & (net.stations_reporting > 0)]
    ref = (base[base.weather_category.isin(["Dégagé", "Nuageux"])]
           .groupby(["day_type", "hour_of_day"]).total_departures.mean().rename("ref"))
    j = base.join(ref, on=["day_type", "hour_of_day"])
    j["ratio"] = j.total_departures / j.ref.replace(0, np.nan)
    parts = []
    for dim, col in [("Météo", "weather_category"), ("Température", "temperature_band")]:
        g = (j.groupby(col).agg(n_hours=("ratio", "size"), avg_departures_per_hour=("total_departures", "mean"),
                                relative_to_reference=("ratio", "mean")).reset_index()
             .rename(columns={col: "category"}))
        g.insert(0, "dimension", dim)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------
# Clustering (même méthode que src/05_ml_clustering.py)
# ---------------------------------------------------------------------------------------------------------

def clustering(sh):
    w = sh[sh.pct_open > .5]
    prof = w.groupby(["station_id", "day_type", "hour_of_day"]).avg_fill_rate.mean().reset_index()
    sat = w.groupby("station_id").agg(pct_empty=("pct_time_empty", "mean"), pct_full=("pct_time_full", "mean"))
    X = prof.pivot_table(index="station_id", columns=["day_type", "hour_of_day"], values="avg_fill_rate")
    X.columns = [f"{'we' if d == 'Week-end' else 'wd'}_fill_{h:02d}" for d, h in X.columns]
    X = X.apply(lambda row: row.fillna(row.mean()), axis=1).dropna().join(sat, how="inner").astype(float)

    results = {}
    for k in range(3, 9):
        pipe = Pipeline([("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=42))])
        labels = pipe.fit_predict(X)
        results[k] = {"silhouette": silhouette_score(pipe["scaler"].transform(X), labels),
                      "inertia": pipe["kmeans"].inertia_, "pipe": pipe, "labels": labels}
    best_k = max(results, key=lambda k: results[k]["silhouette"])
    best = results[best_k]

    assigned = X.assign(cluster_id=best["labels"])
    wd = lambda hours: [f"wd_fill_{h:02d}" for h in hours]  # noqa: E731
    summary = assigned.groupby("cluster_id").agg(n_stations=("pct_empty", "size"), pct_empty=("pct_empty", "mean"),
                                                 pct_full=("pct_full", "mean"))
    summary["early_fill"] = assigned.groupby("cluster_id")[wd(range(5, 8))].mean().mean(axis=1)
    summary["midday_fill"] = assigned.groupby("cluster_id")[wd(range(10, 17))].mean().mean(axis=1)
    raw = {cid: name_cluster(r.early_fill, r.midday_fill, r.pct_empty, r.pct_full) for cid, r in summary.iterrows()}
    summary["cluster_label"] = summary.index.map(dedupe_labels(raw))

    clusters = assigned[["cluster_id"]].reset_index()
    clusters["cluster_label"] = clusters.cluster_id.map(summary.cluster_label)
    clusters["k"], clusters["silhouette"] = best_k, float(best["silhouette"])

    selection = pd.DataFrame([{"k": k, "silhouette": r["silhouette"], "inertia": r["inertia"]}
                              for k, r in results.items()])
    print(f"Clustering : k = {best_k}, silhouette {best['silhouette']:.3f}")
    return clusters, selection


# ---------------------------------------------------------------------------------------------------------
# Prévision (même méthode que src/06_ml_forecast.py)
# ---------------------------------------------------------------------------------------------------------

def add_lags(frame, history):
    """Retards J-1 / J-7 de chaque cible par auto-jointure sur l'heure (comme le SQL du notebook)."""
    out = frame
    for lag in (24, 168):
        lagged = history[["station_id", "hour_local", *TARGETS]].copy()
        lagged["hour_local"] = lagged.hour_local + pd.Timedelta(hours=lag)
        lagged = lagged.rename(columns={t: f"{t}_lag_{lag}h" for t in TARGETS})
        out = out.merge(lagged, on=["station_id", "hour_local"], how="left")
    return out


def prepare(frame, target):
    out = frame.copy()
    out["lag_24h"], out["lag_168h"] = out[f"{target}_lag_24h"], out[f"{target}_lag_168h"]
    for c in ["hour_of_day", "weekday", "is_weekend", "temperature", "precipitation", "wind_speed",
              "lag_24h", "lag_168h", "capacity", "cluster_id"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    if target in out:
        out[target] = out[target].astype(float)
    return out


def train_eval(df, target, cutoff, with_details=False):
    data = prepare(df, target)
    train, test = data[data.hour_local < cutoff], data[data.hour_local >= cutoff]
    train, test = add_station_encodings(train, test, target)
    model = HistGradientBoostingRegressor(**PARAMS).fit(train[FEATURES], train[target])
    pred = model.predict(test[FEATURES])
    baseline = test["lag_168h"].fillna(test["lag_24h"]).fillna(test["station_hour_mean"])
    m, b = regression_metrics(test[target], pred), regression_metrics(test[target], baseline)
    peak = (test.is_weekend.eq(0) & test.hour_of_day.isin(PEAK_HOURS)).to_numpy()
    row = {"target": target, **m, "baseline_mae": b["mae"], "skill_vs_baseline": 1 - m["mae"] / b["mae"],
           "mae_peak_hours": regression_metrics(test.loc[peak, target], pred[peak])["mae"]}
    if not with_details:
        return row, None
    sample = test.sample(min(len(test), 20000), random_state=42)
    imp = permutation_importance(model, sample[FEATURES], sample[target], scoring="neg_mean_absolute_error",
                                 n_repeats=3, random_state=42)
    importance = pd.DataFrame({"target": target, "feature": FEATURES, "importance": imp.importances_mean})
    net = (test.assign(pred=pred, baseline=baseline).groupby("hour_local")[[target, "pred", "baseline"]].sum()
           .reset_index().rename(columns={target: "actual"}))
    net.insert(0, "target", target)
    return row, (importance, net, data)


def forecasting(sh, stations, clusters, weather_future, now_hour):
    h = sh[(sh.n_snapshots >= 2) & (sh.pct_open > .5)]
    df = h[["station_id", "hour_local", "hour_of_day", "weekday", "is_weekend", "temperature", "precipitation",
            "wind_speed", *TARGETS]]
    df = add_lags(df, h)
    df = df.merge(stations[["station_id", "capacity"]], on="station_id", how="left")
    df = df.merge(clusters[["station_id", "cluster_id"]], on="station_id", how="left")

    last_run = now_hour.normalize() + pd.Timedelta(hours=4, minutes=30)
    if last_run > now_hour:
        last_run -= pd.Timedelta(days=1)
    metrics, details = [], {}
    # backtest glissant : un « entraînement nocturne » (4h30) par jour sur les 6 dernières nuits,
    # chacun évalué sur les 7 jours qui précèdent son exécution
    for back in range(5, -1, -1):
        trained_at = last_run - pd.Timedelta(days=back)
        hist = df[df.hour_local < trained_at.floor("h")]
        cutoff = trained_at.normalize() - pd.Timedelta(days=7)
        for target in TARGETS:
            row, det = train_eval(hist, target, cutoff, with_details=back == 0)
            row["trained_at"] = trained_at
            row["mlflow_run_id"] = f"demo-{trained_at:%Y%m%d}-{target}"
            metrics.append(row)
            if det:
                details[target] = det
            print(f"  {trained_at:%d/%m} {target:<10} MAE {row['mae']:.3f}  "
                  f"baseline {row['baseline_mae']:.3f}  skill {row['skill_vs_baseline']:+.1%}")

    # prévision des 24 prochaines heures (modèle ré-entraîné sur tout l'historique)
    st = stations[["station_id", "capacity"]].merge(clusters[["station_id", "cluster_id"]], on="station_id", how="left")
    fut = st.merge(weather_future, how="cross")
    fut["hour_of_day"] = fut.hour_local.dt.hour
    fut["weekday"] = fut.hour_local.dt.dayofweek + 1
    fut["is_weekend"] = (fut.weekday >= 6).astype(int)
    fut = add_lags(fut, h)
    out = fut[["station_id", "hour_local"]].copy()
    for target in TARGETS:
        data = details[target][2]
        full, _ = add_station_encodings(data, data.head(1), target)
        model = HistGradientBoostingRegressor(**PARAMS).fit(full[FEATURES], full[target])
        _, f = add_station_encodings(data, prepare(fut, target), target)
        out[f"pred_{target}"] = np.clip(model.predict(f[FEATURES]), 0, None)
    out["pred_net_flow"] = out.pred_arrivals - out.pred_departures
    out["generated_at"] = now_hour

    importance = pd.concat([d[0] for d in details.values()], ignore_index=True)
    test_net = pd.concat([d[1] for d in details.values()], ignore_index=True)
    return pd.DataFrame(metrics), importance, test_net, out


# ---------------------------------------------------------------------------------------------------------
# État courant et journal qualité
# ---------------------------------------------------------------------------------------------------------

def station_current(sh, dim, now_local):
    last = sh[sh.hour_local == sh.hour_local.max()].set_index("station_id")
    cur = dim[["station_id", "station_label", "lat", "lon", "cluster_label", "capacity"]].copy()
    fill = cur.station_id.map(last.avg_fill_rate).fillna(.5).to_numpy()
    cap = cur.capacity.to_numpy()
    disabled = rng.binomial(cap, .03)
    bikes = np.clip(np.round(fill * cap + rng.normal(0, 1.2, len(cur))), 0, cap - disabled).astype(int)
    cur["available_bikes"] = bikes
    cur["available_stands"] = cap - disabled - bikes
    cur["status"] = np.where(rng.random(len(cur)) < .015, "CLOSED", "OPEN")
    cur["snapshot_local"] = now_local
    cur["usable_capacity"] = cur.available_bikes + cur.available_stands
    cur["fill_rate"] = cur.available_bikes / cur.usable_capacity.replace(0, np.nan)
    cur["availability_status"] = availability_status(cur)
    return cur.drop(columns="capacity")


def availability_status(df):
    return np.select(
        [df.status != "OPEN", df.available_bikes == 0, df.available_stands == 0, df.available_bikes <= 2,
         df.available_stands <= 2],
        ["Fermée", "Vide", "Pleine", "Presque vide", "Presque pleine"], "Disponible")


def dq_log(n_stations, now_utc):
    runs = pd.date_range(end=now_utc.floor("15min"), periods=7 * 96, freq="15min")
    velo_rules = ["station_id_present", "snapshot_ts_present", "station_info_found", "bikes_non_negative",
                  "stands_non_negative", "capacity_positive", "occupancy_consistent", "coords_in_toulouse"]
    meteo_rules = ["time_present", "temperature_plausible", "precipitation_valid", "wind_valid"]
    incident = runs[int(len(runs) * .63)]  # flux station_information incomplet pendant une collecte
    rows = []
    for ts in runs:
        for rule in velo_rules:
            failed = 0
            if rule == "occupancy_consistent":
                failed = int(rng.poisson(.35))
            elif rule == "station_info_found":
                failed = n_stations - 12 if ts == incident else int(rng.random() < .02)
            rows.append((ts, "silver_station_status", rule, failed, n_stations))
        for rule in meteo_rules:
            rows.append((ts, "silver_meteo_hourly", rule, 0, 72))
    return pd.DataFrame(rows, columns=["run_ts", "table_name", "rule", "failed_rows", "total_rows"])


# ---------------------------------------------------------------------------------------------------------

def main(offline: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    now_local = datetime.now(PARIS).replace(tzinfo=None)
    now_hour = pd.Timestamp(now_local).floor("h")
    today = now_hour.normalize()
    start = today - pd.Timedelta(days=HISTORY_DAYS)
    hours = pd.date_range(start, now_hour - pd.Timedelta(hours=1), freq="h")
    all_hours = pd.date_range(start, now_hour + pd.Timedelta(hours=47), freq="h")

    stations = load_stations(offline)
    kinds, dist = assign_archetypes(stations)
    weather_all = simulate_weather(all_hours)
    weather_hist = weather_all[weather_all.hour_local < now_hour].reset_index(drop=True)
    weather_future = weather_all[(weather_all.hour_local >= now_hour)
                                 & (weather_all.hour_local < now_hour + pd.Timedelta(hours=24))]

    print(f"Simulation : {len(stations)} stations × {len(hours)} heures")
    sh = simulate_station_hourly(stations, kinds, dist, hours, weather_hist)

    clusters, k_selection = clustering(sh)

    dim = stations.copy()
    dim["station_label"] = dim.station_id.astype(str).str.zfill(5) + " - " + dim.station_name
    dim = dim.merge(clusters[["station_id", "cluster_id", "cluster_label"]], on="station_id", how="left")
    dim["is_active"] = True

    net = network_hourly(sh)
    profile = saturation_profile(sh, today)
    ranking = saturation_ranking(sh, dim, today)
    effect = weather_effect(net)
    daily = (sh[sh.pct_open > .5].groupby(["station_id", "date_local"])
             .agg(departures=("departures", "sum"), arrivals=("arrivals", "sum"),
                  avg_fill_rate=("avg_fill_rate", "mean"), pct_time_empty=("pct_time_empty", "mean"),
                  pct_time_full=("pct_time_full", "mean")).reset_index())
    cluster_profiles = (sh[sh.pct_open > .5].merge(clusters[["station_id", "cluster_id", "cluster_label"]],
                                                   on="station_id")
                        .groupby(["cluster_id", "cluster_label", "day_type", "hour_of_day"])
                        .agg(avg_fill_rate=("avg_fill_rate", "mean"), avg_departures=("departures", "mean"),
                             avg_arrivals=("arrivals", "mean")).reset_index())

    print("Prévision (backtest glissant + 24 h) :")
    metrics, importance, test_net, forecast = forecasting(sh, stations, clusters, weather_future, now_hour)
    current = station_current(sh, dim, now_local)
    dq = dq_log(len(stations), pd.Timestamp.now(tz="UTC").tz_localize(None))

    tables = {
        "dim_station": dim, "station_current": current, "network_hourly": net, "station_daily": daily,
        "saturation_profile": profile, "saturation_ranking": ranking, "weather_effect": effect,
        "station_clusters": clusters, "cluster_profiles": cluster_profiles,
        "cluster_k_selection": k_selection, "model_metrics": metrics,
        "feature_importance": importance, "forecast_test_network": test_net,
        "forecast_station_hourly": forecast, "dq_log": dq,
        "weather_hourly": weather_all.assign(is_forecast=weather_all.hour_local >= now_hour),
    }
    for name, df in tables.items():
        df.to_parquet(OUT / f"{name}.parquet", index=False)
    (OUT / "meta.json").write_text(json.dumps({
        "generated_at": now_local.strftime("%Y-%m-%d %H:%M:%S"),
        "history_start": f"{start:%Y-%m-%d}",
        "n_stations": int(len(stations)),
        "n_station_hours": int(len(sh)),
        "seed": SEED,
    }, indent=2), encoding="utf-8")
    size = sum(f.stat().st_size for f in OUT.glob("*")) / 1e6
    print(f"{len(tables)} tables écrites dans {OUT.relative_to(ROOT)} ({size:.1f} Mo)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true", help="stations synthétiques, sans appel au flux GBFS")
    main(ap.parse_args().offline)
