# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Silver → Gold
# MAGIC Tables métier consommées par Power BI, les modèles ML et les recommandations.
# MAGIC
# MAGIC | Table | Grain | Usage |
# MAGIC |---|---|---|
# MAGIC | `gold_station_hourly` | station × heure | fait principal : fréquentation, remplissage, saturation, météo |
# MAGIC | `gold_network_hourly` | heure | vue réseau, effet météo |
# MAGIC | `gold_dim_station` | station | dimension (nom, position, capacité, cluster) |
# MAGIC | `gold_station_current` | station | disponibilité en temps réel |
# MAGIC | `gold_saturation_profile` | station × type de jour × heure | profils sur 28 jours |
# MAGIC | `gold_saturation_ranking` | station | classement des stations saturées aux heures de pointe |
# MAGIC | `gold_weather_effect` | catégorie météo / tranche de température | effet sur les départs |
# MAGIC
# MAGIC **Estimation des flux.** L'API ne publie pas les trajets : départs et arrivées sont estimés à partir des
# MAGIC variations du nombre de vélos entre deux collectes successives (≤ 30 min d'écart). C'est un **minorant**
# MAGIC (un départ et une arrivée dans le même intervalle s'annulent) qui inclut aussi les opérations de
# MAGIC rééquilibrage par camion.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "mobility")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
T = f"{catalog}.{schema}"
spark.conf.set("spark.sql.session.timeZone", "UTC")

PEAK_FILTER = "day_type = 'Semaine' AND hour_of_day IN (7, 8, 9, 17, 18, 19)"

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {T}.gold_station_hourly
CLUSTER BY (hour_local, station_id)
COMMENT 'Indicateurs horaires par station, croisés avec la météo'
AS
WITH snaps AS (
  SELECT
    station_id, snapshot_ts, snapshot_local, available_bikes, available_stands, status,
    available_bikes - LAG(available_bikes) OVER w AS delta_bikes,
    (unix_timestamp(snapshot_ts) - unix_timestamp(LAG(snapshot_ts) OVER w)) / 60.0 AS gap_min
  FROM {T}.silver_station_status
  WINDOW w AS (PARTITION BY station_id ORDER BY snapshot_ts)
),
hourly AS (
  SELECT
    station_id,
    date_trunc('HOUR', snapshot_local)                                         AS hour_local,
    count(*)                                                                   AS n_snapshots,
    avg(CASE WHEN status = 'OPEN' THEN 1D ELSE 0D END)                       AS pct_open,
    avg(available_bikes)                                                       AS avg_bikes,
    min(available_bikes)                                                       AS min_bikes,
    max(available_bikes)                                                       AS max_bikes,
    avg(available_bikes / nullif(available_bikes + available_stands, 0))       AS avg_fill_rate,
    avg(CASE WHEN status = 'OPEN' AND available_bikes = 0  THEN 1D ELSE 0D END) AS pct_time_empty,
    avg(CASE WHEN status = 'OPEN' AND available_stands = 0 THEN 1D ELSE 0D END) AS pct_time_full,
    sum(CASE WHEN gap_min <= 30 AND delta_bikes < 0 THEN -delta_bikes ELSE 0 END) AS departures,
    sum(CASE WHEN gap_min <= 30 AND delta_bikes > 0 THEN  delta_bikes ELSE 0 END) AS arrivals
  FROM snaps
  GROUP BY ALL
)
SELECT
  h.*,
  h.arrivals - h.departures                          AS net_flow,
  to_date(h.hour_local)                              AS date_local,
  hour(h.hour_local)                                 AS hour_of_day,
  ((dayofweek(h.hour_local) + 5) % 7) + 1            AS weekday,      -- 1 = lundi
  CASE WHEN dayofweek(h.hour_local) IN (1, 7) THEN 'Week-end' ELSE 'Semaine' END AS day_type,
  CASE WHEN dayofweek(h.hour_local) IN (1, 7) THEN 1 ELSE 0 END                  AS is_weekend,
  m.temperature_2m   AS temperature,
  m.precipitation,
  m.wind_speed_10m   AS wind_speed,
  m.weather_code,
  m.weather_category
FROM hourly h
LEFT JOIN {T}.silver_meteo_hourly m ON m.time_local = h.hour_local
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {T}.gold_network_hourly
COMMENT 'Indicateurs horaires à l échelle du réseau'
AS
SELECT
  hour_local, date_local, hour_of_day, weekday, day_type, is_weekend,
  sum(departures)                                        AS total_departures,
  sum(arrivals)                                          AS total_arrivals,
  avg(avg_fill_rate)                                     AS avg_fill_rate,
  sum(CASE WHEN pct_time_empty >= 0.5 THEN 1 ELSE 0 END) AS stations_mostly_empty,
  sum(CASE WHEN pct_time_full  >= 0.5 THEN 1 ELSE 0 END) AS stations_mostly_full,
  count(*)                                               AS stations_reporting,
  first(temperature, true)                               AS temperature,
  first(precipitation, true)                             AS precipitation,
  first(wind_speed, true)                                AS wind_speed,
  first(weather_category, true)                          AS weather_category,
  CASE WHEN first(temperature, true) < 5  THEN '1. < 5 °C'
       WHEN first(temperature, true) < 15 THEN '2. 5-15 °C'
       WHEN first(temperature, true) < 25 THEN '3. 15-25 °C'
       WHEN first(temperature, true) IS NOT NULL THEN '4. > 25 °C' END AS temperature_band
FROM {T}.gold_station_hourly
GROUP BY ALL
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dimension station (avec le segment issu du clustering s'il existe)

# COMMAND ----------

has_clusters = spark.catalog.tableExists(f"{T}.gold_station_clusters")
cluster_cols = "c.cluster_id, c.cluster_label" if has_clusters else "CAST(NULL AS INT) AS cluster_id, CAST(NULL AS STRING) AS cluster_label"
cluster_join = f"LEFT JOIN {T}.gold_station_clusters c USING (station_id)" if has_clusters else ""

spark.sql(f"""
CREATE OR REPLACE TABLE {T}.gold_dim_station
COMMENT 'Dimension station'
AS
SELECT s.station_id,
       concat(lpad(CAST(s.station_id AS STRING), 5, '0'), ' - ', s.station_name) AS station_label,
       s.station_name, s.address, s.lat, s.lon, s.capacity, s.banking,
       s.first_seen, s.last_seen,
       s.last_seen >= current_timestamp() - INTERVAL 1 DAY AS is_active,
       {cluster_cols}
FROM {T}.silver_stations s
{cluster_join}
""")

spark.sql(f"""
CREATE OR REPLACE TABLE {T}.gold_station_current
COMMENT 'Dernier état connu de chaque station'
AS
WITH last AS (
  SELECT *, row_number() OVER (PARTITION BY station_id ORDER BY snapshot_ts DESC) AS rn
  FROM {T}.silver_station_status
)
SELECT l.station_id, d.station_label, d.lat, d.lon, d.cluster_label,
       l.snapshot_local, l.status, l.available_bikes, l.available_stands,
       l.available_bikes + l.available_stands AS usable_capacity,
       l.fill_rate,
       CASE WHEN l.status <> 'OPEN'          THEN 'Fermée'
            WHEN l.available_bikes = 0       THEN 'Vide'
            WHEN l.available_stands = 0      THEN 'Pleine'
            WHEN l.available_bikes <= 2      THEN 'Presque vide'
            WHEN l.available_stands <= 2     THEN 'Presque pleine'
            ELSE 'Disponible' END             AS availability_status
FROM last l
JOIN {T}.gold_dim_station d USING (station_id)
WHERE l.rn = 1
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saturation

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {T}.gold_saturation_profile
COMMENT 'Profil moyen par station, type de jour et heure (28 derniers jours)'
AS
SELECT station_id, day_type, hour_of_day,
       count(*)            AS n_hours,
       avg(avg_fill_rate)  AS avg_fill_rate,
       avg(pct_time_empty) AS pct_time_empty,
       avg(pct_time_full)  AS pct_time_full,
       avg(departures)     AS avg_departures,
       avg(arrivals)       AS avg_arrivals,
       avg(net_flow)       AS avg_net_flow,
       CASE WHEN avg(pct_time_empty) >= 0.25 THEN 'Souvent vide'
            WHEN avg(pct_time_full)  >= 0.25 THEN 'Souvent pleine'
            ELSE 'Normale' END AS saturation_status
FROM {T}.gold_station_hourly
WHERE date_local >= date_sub(current_date(), 28) AND pct_open > 0.5
GROUP BY ALL
""")

spark.sql(f"""
CREATE OR REPLACE TABLE {T}.gold_saturation_ranking
COMMENT 'Stations les plus souvent vides ou pleines aux heures de pointe de semaine (28 jours)'
AS
WITH peak AS (
  SELECT station_id,
         avg(pct_time_empty) AS pct_empty_peak,
         avg(pct_time_full)  AS pct_full_peak,
         avg(CASE WHEN hour_of_day < 12 THEN net_flow END) AS avg_net_flow_morning,
         avg(CASE WHEN hour_of_day >= 12 THEN net_flow END) AS avg_net_flow_evening
  FROM {T}.gold_station_hourly
  WHERE date_local >= date_sub(current_date(), 28) AND {PEAK_FILTER} AND pct_open > 0.5
  GROUP BY station_id
)
SELECT p.*, d.station_label, d.lat, d.lon, d.cluster_label,
       p.pct_empty_peak + p.pct_full_peak                                  AS pct_saturated_peak,
       dense_rank() OVER (ORDER BY p.pct_empty_peak + p.pct_full_peak DESC) AS saturation_rank,
       CASE WHEN p.pct_empty_peak >= p.pct_full_peak THEN 'Manque de vélos' ELSE 'Manque de bornes' END AS main_issue
FROM peak p JOIN {T}.gold_dim_station d USING (station_id)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Effet de la météo
# MAGIC Pour neutraliser l'effet horaire, chaque heure est comparée à la moyenne des heures **sèches** de même
# MAGIC type de jour et de même heure : `relative_to_reference = 0.8` signifie 20 % de départs en moins.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {T}.gold_weather_effect
COMMENT 'Départs relatifs par condition météo et tranche de température, à heure comparable'
AS
WITH base AS (
  SELECT * FROM {T}.gold_network_hourly
  WHERE weather_category IS NOT NULL AND stations_reporting > 0
),
ref AS (
  SELECT day_type, hour_of_day, avg(total_departures) AS ref_departures
  FROM base WHERE weather_category IN ('Dégagé', 'Nuageux')
  GROUP BY ALL
),
joined AS (
  SELECT b.*, b.total_departures / nullif(r.ref_departures, 0) AS ratio
  FROM base b JOIN ref r USING (day_type, hour_of_day)
)
SELECT 'Météo' AS dimension, weather_category AS category,
       count(*) AS n_hours, avg(total_departures) AS avg_departures_per_hour, avg(ratio) AS relative_to_reference
FROM joined GROUP BY ALL
UNION ALL
SELECT 'Température', temperature_band, count(*), avg(total_departures), avg(ratio)
FROM joined WHERE temperature_band IS NOT NULL GROUP BY ALL
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Contrôles qualité gold
# MAGIC Unicité du grain, valeurs non négatives, couverture météo. Un échec fait échouer la tâche.

# COMMAND ----------

checks = spark.sql(f"""
SELECT
  (SELECT count(*) - count(DISTINCT station_id, hour_local) FROM {T}.gold_station_hourly) AS duplicate_station_hours,
  (SELECT count(*) FROM {T}.gold_station_hourly WHERE departures < 0 OR arrivals < 0)     AS negative_flows,
  (SELECT count(*) FROM {T}.gold_station_hourly WHERE avg_fill_rate NOT BETWEEN 0 AND 1)   AS fill_rate_out_of_range,
  (SELECT avg(CASE WHEN temperature IS NULL THEN 1D ELSE 0D END)
     FROM {T}.gold_station_hourly WHERE date_local < current_date())                      AS share_missing_weather
""").first().asDict()

print(checks)
errors = [k for k in ("duplicate_station_hours", "negative_flows", "fill_rate_out_of_range") if checks[k]]
if (checks["share_missing_weather"] or 0) > 0.05:
    errors.append("share_missing_weather > 5 %")
if errors:
    raise RuntimeError(f"Contrôles gold en échec : {errors}")
