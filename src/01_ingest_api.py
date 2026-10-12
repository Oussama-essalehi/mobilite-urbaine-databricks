# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingestion API → zone landing
# MAGIC Appelé par le job planifié. À chaque exécution :
# MAGIC 1. récupère l'état temps réel de **toutes** les stations VélôToulouse (open data Toulouse Métropole) ;
# MAGIC 2. récupère la météo horaire de Toulouse (Open-Meteo, sans clé d'API) ;
# MAGIC 3. écrit chaque réponse **telle quelle** en JSON dans le volume `landing`, partitionnée par date.
# MAGIC
# MAGIC Aucune transformation ici : le brut est conservé pour pouvoir rejouer le pipeline.

# COMMAND ----------

import json
import time
from datetime import datetime, timezone

import requests

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "mobility")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
LANDING = f"/Volumes/{catalog}/{schema}/landing"

# Toulouse centre — la météo est homogène à l'échelle de la métropole
LAT, LON = 43.6047, 1.4442

SOURCES = {
    "velo": {
        # endpoint "exports" : renvoie tous les enregistrements sans pagination (~290 stations)
        "url": "https://data.toulouse-metropole.fr/api/explore/v2.1/catalog/datasets/"
               "api-velo-toulouse-temps-reel/exports/json",
        "params": {},
    },
    "meteo": {
        "url": "https://api.open-meteo.com/v1/forecast",
        "params": {
            "latitude": LAT,
            "longitude": LON,
            "timezone": "Europe/Paris",
            "past_days": 1,       # recouvrement volontaire : les doublons sont gérés en silver
            "forecast_days": 2,   # couvre toujours les 24 prochaines heures (utilisé par la prévision)
            "hourly": ",".join([
                "temperature_2m", "apparent_temperature", "precipitation", "rain",
                "relative_humidity_2m", "wind_speed_10m", "weather_code", "is_day",
            ]),
        },
    },
}

# COMMAND ----------

def fetch(url: str, params: dict, retries: int = 3, backoff: float = 5.0):
    """GET avec reprises (backoff exponentiel) — une API open data peut être lente ou indisponible."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[{attempt}/{retries}] échec {url} : {e}")
            if attempt < retries:
                time.sleep(backoff * 2 ** (attempt - 1))
    raise RuntimeError(f"Source indisponible après {retries} tentatives : {url}") from last_err


def write_landing(source: str, payload, fetched_at: datetime) -> str:
    folder = f"{LANDING}/{source}/date={fetched_at:%Y-%m-%d}"
    dbutils.fs.mkdirs(folder)
    path = f"{folder}/{source}_{fetched_at:%Y%m%dT%H%M%SZ}.json"
    envelope = {
        "source": source,
        "fetched_at": fetched_at.isoformat(),
        "payload": payload,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, ensure_ascii=False)
    return path

# COMMAND ----------

fetched_at = datetime.now(timezone.utc).replace(microsecond=0)
errors = []

for name, cfg in SOURCES.items():
    try:
        payload = fetch(cfg["url"], cfg["params"])
        if name == "velo" and (not isinstance(payload, list) or len(payload) == 0):
            raise ValueError("réponse vélo vide ou inattendue")
        path = write_landing(name, payload, fetched_at)
        size = len(payload) if isinstance(payload, list) else "-"
        print(f"OK  {name:<6} → {path}  (enregistrements : {size})")
    except Exception as e:  # noqa: BLE001
        errors.append(f"{name}: {e}")

# Faire échouer la tâche si une source a échoué → le job envoie l'alerte e-mail
if errors:
    raise RuntimeError("Ingestion incomplète — " + " | ".join(errors))
