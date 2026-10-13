"""Accès aux données du tableau de bord.

Deux sources, même contrat de tables (noms et colonnes des tables gold) :

- **databricks** : requêtes sur le SQL warehouse (`databricks-sql-connector`), activé dès que les
  identifiants sont fournis dans `.streamlit/secrets.toml` (section `[databricks]`) ou en variables
  d'environnement `DATABRICKS_SERVER_HOSTNAME`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_TOKEN` ;
- **demo** : instantané Parquet produit par `scripts/build_demo_data.py` (historique simulé).

Les tables qui n'existent que dans MLflow (sélection de k, ACP, importance des variables, série de test) ne
sont disponibles qu'en démo : les pages concernées affichent alors un message à la place du graphique.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo_data"

# Requêtes Databricks : agrégées côté warehouse pour ne transférer que ce que les pages affichent.
QUERIES = {
    "dim_station": "SELECT * FROM {T}.gold_dim_station",
    "station_current": "SELECT * FROM {T}.gold_station_current",
    "network_hourly": "SELECT * FROM {T}.gold_network_hourly WHERE date_local >= date_sub(current_date(), 60)",
    "station_daily": """
        SELECT station_id, date_local, sum(departures) AS departures, sum(arrivals) AS arrivals,
               avg(avg_fill_rate) AS avg_fill_rate, avg(pct_time_empty) AS pct_time_empty,
               avg(pct_time_full) AS pct_time_full
        FROM {T}.gold_station_hourly
        WHERE date_local >= date_sub(current_date(), 60) AND pct_open > 0.5
        GROUP BY ALL""",
    "saturation_profile": "SELECT * FROM {T}.gold_saturation_profile",
    "saturation_ranking": "SELECT * FROM {T}.gold_saturation_ranking",
    "weather_effect": "SELECT * FROM {T}.gold_weather_effect",
    "station_clusters": "SELECT * FROM {T}.gold_station_clusters",
    "cluster_profiles": """
        SELECT c.cluster_id, c.cluster_label, h.day_type, h.hour_of_day,
               avg(h.avg_fill_rate) AS avg_fill_rate, avg(h.departures) AS avg_departures,
               avg(h.arrivals) AS avg_arrivals
        FROM {T}.gold_station_hourly h JOIN {T}.gold_station_clusters c USING (station_id)
        WHERE h.date_local >= date_sub(current_date(), 56) AND h.pct_open > 0.5
        GROUP BY ALL""",
    "forecast_station_hourly": "SELECT * FROM {T}.gold_forecast_station_hourly",
    "model_metrics": "SELECT * FROM {T}.gold_model_metrics ORDER BY trained_at",
    "dq_log": "SELECT * FROM {T}.dq_log WHERE run_ts >= current_timestamp() - INTERVAL 7 DAYS",
    "weather_hourly": """
        SELECT time_local AS hour_local, temperature_2m AS temperature, precipitation,
               wind_speed_10m AS wind_speed, weather_code, weather_category, is_forecast
        FROM {T}.silver_meteo_hourly WHERE time_local >= current_date() - INTERVAL 60 DAYS""",
}


@dataclass(frozen=True)
class Source:
    mode: str           # "databricks" | "demo"
    label: str
    detail: str
    error: str | None = None   # échec de connexion Databricks (bascule sur la démo)


def _databricks_config() -> dict | None:
    cfg = {}
    try:
        if "databricks" in st.secrets:
            cfg = dict(st.secrets["databricks"])
    except Exception:  # noqa: BLE001 — pas de secrets.toml
        cfg = {}
    cfg.setdefault("server_hostname", os.getenv("DATABRICKS_SERVER_HOSTNAME"))
    cfg.setdefault("http_path", os.getenv("DATABRICKS_HTTP_PATH"))
    cfg.setdefault("access_token", os.getenv("DATABRICKS_TOKEN"))
    cfg.setdefault("catalog", os.getenv("DATABRICKS_CATALOG", "workspace"))
    cfg.setdefault("schema", os.getenv("DATABRICKS_SCHEMA", "mobility"))
    if all(cfg.get(k) for k in ("server_hostname", "http_path", "access_token")):
        return cfg
    return None


def _naive(df: pd.DataFrame) -> pd.DataFrame:
    """Le warehouse renvoie des timestamps UTC qui portent l'heure locale (session en UTC) : on retire le fuseau
    sans convertir pour retrouver l'heure de Paris stockée dans les tables."""
    for c in df.columns:
        if isinstance(df[c].dtype, pd.DatetimeTZDtype):
            df[c] = df[c].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


@st.cache_data(ttl=900, show_spinner="Requête sur le SQL warehouse Databricks…")
def _query_databricks(name: str, cfg_items: tuple) -> pd.DataFrame | None:
    from databricks import sql

    cfg = dict(cfg_items)
    T = f"{cfg['catalog']}.{cfg['schema']}"
    try:
        with sql.connect(server_hostname=cfg["server_hostname"], http_path=cfg["http_path"],
                         access_token=cfg["access_token"],
                         session_configuration={"spark.sql.session.timeZone": "UTC"}) as conn:
            with conn.cursor() as cur:
                cur.execute(QUERIES[name].format(T=T))
                return _naive(cur.fetchall_arrow().to_pandas())
    except Exception as e:  # noqa: BLE001 — table absente (ML pas encore exécuté) ou erreur de requête
        if "TABLE_OR_VIEW_NOT_FOUND" in str(e):
            return None
        raise


@st.cache_data(show_spinner=False)
def _read_demo(name: str) -> pd.DataFrame | None:
    path = DEMO_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def demo_meta() -> dict:
    path = DEMO_DIR / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_resource(show_spinner=False)
def _probe_databricks(cfg_items: tuple) -> str | None:
    """Teste la connexion une fois par session serveur ; renvoie le message d'erreur éventuel."""
    try:
        _query_databricks("dim_station", cfg_items)
        return None
    except Exception as e:  # noqa: BLE001
        return str(e)[:300]


def source() -> Source:
    cfg = _databricks_config()
    err = None
    if cfg:
        err = _probe_databricks(tuple(sorted(cfg.items())))
        if err is None:
            return Source("databricks", "Databricks · production",
                          f"SQL warehouse · {cfg['catalog']}.{cfg['schema']}")
    meta = demo_meta()
    return Source("demo", "Instantané de démonstration",
                  f"Historique simulé · généré le {meta.get('generated_at', '?')[:16]}", error=err)


def load(name: str) -> pd.DataFrame | None:
    """Table `name` depuis la source active, ou None si elle n'existe pas (encore)."""
    cfg = _databricks_config()
    if cfg and source().mode == "databricks":
        if name not in QUERIES:
            return None
        df = _query_databricks(name, tuple(sorted(cfg.items())))
    else:
        df = _read_demo(name)
    return None if df is None else df.copy()


def reference_time() -> pd.Timestamp:
    """Instant « maintenant » des analyses : heure de Paris en direct, heure de génération en démo."""
    if source().mode == "databricks":
        return pd.Timestamp.now(tz="Europe/Paris").tz_localize(None)
    return pd.Timestamp(demo_meta().get("generated_at", pd.Timestamp.now()))
