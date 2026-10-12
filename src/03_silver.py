# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Bronze → Silver
# MAGIC Lecture **incrémentale** des tables bronze (Structured Streaming `availableNow` + `foreachBatch`) :
# MAGIC
# MAGIC | Table | Grain | Écriture |
# MAGIC |---|---|---|
# MAGIC | `silver_station_status` | station × instant de collecte | MERGE (idempotent) |
# MAGIC | `silver_stations` | station (attributs les plus récents) | MERGE (SCD 1) |
# MAGIC | `silver_meteo_hourly` | heure locale | MERGE (la mesure la plus récente gagne) |
# MAGIC | `silver_velo_quarantine` | lignes rejetées + règles en échec | append |
# MAGIC | `dq_log` | règle × exécution | append |
# MAGIC
# MAGIC Les contrôles qualité sont déclarés comme des expectations (expressions SQL). Une ligne invalide part en
# MAGIC quarantaine ; si plus de 20 % d'un lot est invalide, ou si les données ne sont plus fraîches, la tâche échoue
# MAGIC et le job envoie une alerte.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.getcwd())

from delta.tables import DeltaTable
from pyspark.sql import Window
from pyspark.sql import functions as F

from mobility.quality import log_quality, split_by_expectations, summarize

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "mobility")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
T = f"{catalog}.{schema}"
CHECKPOINTS = f"/Volumes/{catalog}/{schema}/checkpoints"

spark.conf.set("spark.sql.session.timeZone", "UTC")

MAX_INVALID_SHARE = 0.20
MAX_STALENESS_MIN = 60

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tables cibles

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {T}.silver_station_status (
  station_id        INT       COMMENT 'Numéro de station VélôToulouse',
  snapshot_ts       TIMESTAMP COMMENT 'Instant de collecte (UTC)',
  snapshot_local    TIMESTAMP COMMENT 'Instant de collecte, heure de Paris',
  available_bikes   INT,
  available_stands  INT,
  capacity          INT       COMMENT 'Nombre total de bornes',
  fill_rate         DOUBLE    COMMENT 'vélos / (vélos + bornes libres)',
  status            STRING    COMMENT 'OPEN / CLOSED',
  last_update       TIMESTAMP COMMENT 'Dernière mise à jour déclarée par l opérateur',
  processed_at      TIMESTAMP
) CLUSTER BY (station_id, snapshot_ts)
COMMENT 'État des stations, une ligne par station et par collecte'
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {T}.silver_stations (
  station_id   INT,
  station_name STRING,
  address      STRING,
  lat          DOUBLE,
  lon          DOUBLE,
  capacity     INT,
  banking      BOOLEAN COMMENT 'Terminal de paiement',
  first_seen   TIMESTAMP,
  last_seen    TIMESTAMP
) COMMENT 'Référentiel des stations (attributs les plus récents)'
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {T}.silver_meteo_hourly (
  time_local           TIMESTAMP COMMENT 'Heure locale (Europe/Paris)',
  temperature_2m       DOUBLE,
  apparent_temperature DOUBLE,
  precipitation        DOUBLE COMMENT 'mm',
  rain                 DOUBLE COMMENT 'mm',
  relative_humidity_2m DOUBLE,
  wind_speed_10m       DOUBLE COMMENT 'km/h',
  weather_code         INT    COMMENT 'Code WMO',
  is_day               INT,
  weather_category     STRING,
  is_forecast          BOOLEAN COMMENT 'TRUE si la valeur est une prévision au moment de la collecte',
  fetched_at           TIMESTAMP
) COMMENT 'Météo horaire de Toulouse'
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vélos — parsing & qualité
# MAGIC Les enregistrements sont lus comme `map<string,string>` puis typés avec `try_cast` : un champ manquant ou mal
# MAGIC formé devient NULL (et part en quarantaine) au lieu de faire planter le pipeline. Les noms de champs
# MAGIC alternatifs (`number` / `station_number`, `position` / `geo_point_2d`…) sont gérés par `coalesce`.

# COMMAND ----------

VELO_RULES = {
    "station_id_present":   "station_id IS NOT NULL",
    "snapshot_ts_present":  "snapshot_ts IS NOT NULL",
    "bikes_non_negative":   "available_bikes >= 0",
    "stands_non_negative":  "available_stands >= 0",
    "capacity_positive":    "capacity > 0",
    "occupancy_consistent": "available_bikes + available_stands <= capacity",
    "coords_in_toulouse":   "lat BETWEEN 43.45 AND 43.75 AND lon BETWEEN 1.25 AND 1.65",
    "status_known":         "status IN ('OPEN', 'CLOSED')",
}


def parse_velo(bronze_df):
    records = bronze_df.selectExpr(
        "fetched_at",
        "explode(from_json(get_json_object(raw_json, '$.payload'), 'array<map<string,string>>')) AS m",
    )
    return records.selectExpr(
        "fetched_at AS snapshot_ts",
        "from_utc_timestamp(fetched_at, 'Europe/Paris') AS snapshot_local",
        "try_cast(coalesce(m['number'], m['station_number']) AS INT) AS station_id",
        "initcap(trim(regexp_replace(m['name'], '^[0-9]+\\\\s*-\\\\s*', ''))) AS station_name",
        "initcap(trim(m['address'])) AS address",
        "try_cast(coalesce(m['bike_stands'], m['capacity']) AS INT) AS capacity",
        "try_cast(coalesce(m['available_bikes'], m['bikes']) AS INT) AS available_bikes",
        "try_cast(coalesce(m['available_bike_stands'], m['stands']) AS INT) AS available_stands",
        "upper(m['status']) AS status",
        "coalesce(m['position'], m['geo_point_2d']) AS pos",
        "m['lat'] AS lat_raw", "coalesce(m['lon'], m['lng']) AS lon_raw",
        "lower(m['banking']) IN ('true', '1', 'oui') AS banking",
        "coalesce(try_to_timestamp(m['last_update']), "
        "timestamp_millis(try_cast(m['last_update'] AS BIGINT))) AS last_update",
    ).selectExpr(
        "*",
        "try_cast(coalesce(get_json_object(pos, '$.lat'), lat_raw) AS DOUBLE) AS lat",
        "try_cast(coalesce(get_json_object(pos, '$.lon'), get_json_object(pos, '$.lng'), lon_raw) AS DOUBLE) AS lon",
    ).drop("pos", "lat_raw", "lon_raw")


STATUS_COLS = ["station_id", "snapshot_ts", "snapshot_local", "available_bikes", "available_stands",
               "capacity", "fill_rate", "status", "last_update", "processed_at"]


def process_velo_batch(batch_df, batch_id):
    sp = batch_df.sparkSession
    parsed = parse_velo(batch_df)

    counts = summarize(parsed, VELO_RULES)
    log_quality(sp, f"{T}.dq_log", "silver_station_status", counts)
    total = counts["__total"]
    if total == 0:
        return

    valid, invalid = split_by_expectations(parsed, VELO_RULES)
    (invalid.withColumn("quarantined_at", F.current_timestamp())
        .write.mode("append").option("mergeSchema", "true")
        .saveAsTable(f"{T}.silver_velo_quarantine"))

    # dédoublonnage intra-lot (rejeu d'un même fichier, etc.)
    w = Window.partitionBy("station_id", "snapshot_ts").orderBy(F.col("last_update").desc_nulls_last())
    valid = (valid.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")
             .withColumn("fill_rate", F.expr("available_bikes / nullif(available_bikes + available_stands, 0)"))
             .withColumn("processed_at", F.current_timestamp()))

    (DeltaTable.forName(sp, f"{T}.silver_station_status").alias("t")
        .merge(valid.select(*STATUS_COLS).alias("s"),
               "t.station_id = s.station_id AND t.snapshot_ts = s.snapshot_ts")
        .whenNotMatchedInsertAll()
        .execute())

    # référentiel des stations : dernière version connue de chaque station
    wl = Window.partitionBy("station_id").orderBy(F.col("snapshot_ts").desc())
    latest = (valid.withColumn("_rn", F.row_number().over(wl)).filter("_rn = 1")
              .selectExpr("station_id", "station_name", "address", "lat", "lon", "capacity", "banking",
                          "snapshot_ts AS first_seen", "snapshot_ts AS last_seen"))
    (DeltaTable.forName(sp, f"{T}.silver_stations").alias("t")
        .merge(latest.alias("s"), "t.station_id = s.station_id")
        .whenMatchedUpdate(
            condition="s.last_seen >= t.last_seen",
            set={c: f"s.{c}" for c in ["station_name", "address", "lat", "lon", "capacity", "banking", "last_seen"]})
        .whenNotMatchedInsertAll()
        .execute())

    failed = sum(v for k, v in counts.items() if k != "__total")
    n_invalid = invalid.count()
    if n_invalid / total > MAX_INVALID_SHARE:
        raise RuntimeError(f"Lot {batch_id} : {n_invalid}/{total} lignes invalides "
                           f"({failed} échecs de règles) — voir silver_velo_quarantine")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Météo — parsing & qualité

# COMMAND ----------

METEO_VARS = ["temperature_2m", "apparent_temperature", "precipitation", "rain",
              "relative_humidity_2m", "wind_speed_10m", "weather_code", "is_day"]

METEO_RULES = {
    "time_present":         "time_local IS NOT NULL",
    "temperature_plausible": "temperature_2m BETWEEN -25 AND 50",
    "precipitation_valid":  "precipitation >= 0",
    "wind_valid":           "wind_speed_10m >= 0",
}


def parse_meteo(bronze_df):
    hourly_schema = "struct<time:array<string>," + ",".join(f"{v}:array<double>" for v in METEO_VARS) + ">"
    fields = ", ".join(f"'{v}', try_element_at(h.{v}, i)" for v in METEO_VARS)
    return (
        bronze_df.selectExpr(
            "fetched_at",
            f"from_json(get_json_object(raw_json, '$.payload.hourly'), '{hourly_schema}') AS h")
        .selectExpr(
            "fetched_at",
            f"explode(transform(sequence(1, size(h.time)), i -> "
            f"named_struct('time', try_element_at(h.time, i), {fields}))) AS r")
        .selectExpr(
            "try_to_timestamp(r.time, \"yyyy-MM-dd'T'HH:mm\") AS time_local",
            *[f"r.{v} AS {v}" for v in METEO_VARS if v not in ("weather_code", "is_day")],
            "CAST(r.weather_code AS INT) AS weather_code",
            "CAST(r.is_day AS INT) AS is_day",
            "fetched_at")
        .selectExpr(
            "*",
            """CASE WHEN precipitation >= 2 THEN 'Forte pluie'
                    WHEN precipitation > 0.1 THEN 'Pluie'
                    WHEN weather_code IN (0, 1) THEN 'Dégagé'
                    ELSE 'Nuageux' END AS weather_category""",
            "time_local > from_utc_timestamp(fetched_at, 'Europe/Paris') AS is_forecast")
    )


def process_meteo_batch(batch_df, batch_id):
    sp = batch_df.sparkSession
    parsed = parse_meteo(batch_df)
    log_quality(sp, f"{T}.dq_log", "silver_meteo_hourly", summarize(parsed, METEO_RULES))
    valid, _ = split_by_expectations(parsed, METEO_RULES)

    w = Window.partitionBy("time_local").orderBy(F.col("fetched_at").desc())
    latest = valid.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")

    (DeltaTable.forName(sp, f"{T}.silver_meteo_hourly").alias("t")
        .merge(latest.alias("s"), "t.time_local = s.time_local")
        .whenMatchedUpdateAll(condition="s.fetched_at > t.fetched_at")
        .whenNotMatchedInsertAll()
        .execute())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exécution incrémentale

# COMMAND ----------

for source, fn in [("velo", process_velo_batch), ("meteo", process_meteo_batch)]:
    (spark.readStream.table(f"{T}.bronze_{source}_raw")
        .writeStream
        .foreachBatch(fn)
        .option("checkpointLocation", f"{CHECKPOINTS}/silver_{source}")
        .trigger(availableNow=True)
        .start()
        .awaitTermination())
    print(f"silver {source} : OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Contrôle de fraîcheur

# COMMAND ----------

staleness = spark.sql(f"""
    SELECT (unix_timestamp(current_timestamp()) - unix_timestamp(max(snapshot_ts))) / 60 AS minutes
    FROM {T}.silver_station_status
""").first()["minutes"]

print(f"Dernière collecte il y a {staleness:.0f} min" if staleness is not None else "Aucune donnée")
if staleness is None or staleness > MAX_STALENESS_MIN:
    raise RuntimeError(f"Données vélo non fraîches (> {MAX_STALENESS_MIN} min)")
