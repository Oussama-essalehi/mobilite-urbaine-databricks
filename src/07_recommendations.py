# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Recommandations de rééquilibrage
# MAGIC Avant la **prochaine pointe** (semaine 7h-10h / 17h-20h, week-end 14h-19h) :
# MAGIC
# MAGIC 1. on projette le nombre de vélos de chaque station à la fin de la pointe :
# MAGIC    `vélos actuels + flux net attendu d'ici là` — flux issu de la **prévision ML** si elle couvre la fenêtre,
# MAGIC    sinon du **profil historique** (28 jours) ;
# MAGIC 2. une station passe sous le seuil bas (≥ 2 vélos ou 15 % de la capacité) → **à réapprovisionner** ;
# MAGIC    au-dessus du seuil haut → **à délester** ; la quantité ramène la station à 50 % de remplissage ;
# MAGIC 3. les stations à délester sont appariées aux stations à réapprovisionner les plus proches (≤ 3 km),
# MAGIC    par priorité (quantité × fréquence de saturation historique sur ce créneau).

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.getcwd())

import pandas as pd
from pyspark.sql import functions as F

from mobility.rebalancing import classify_station, next_peak_window, pair_moves

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "mobility")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
T = f"{catalog}.{schema}"
spark.conf.set("spark.sql.session.timeZone", "UTC")

MAX_PAIR_KM = 3.0

# COMMAND ----------

from datetime import datetime

# lu en texte : la conversion Spark → Python d'un timestamp dépend du fuseau de la machine
now_local = datetime.strptime(
    spark.sql("SELECT date_format(from_utc_timestamp(current_timestamp(), 'Europe/Paris'), "
              "'yyyy-MM-dd HH:mm:ss') AS t").first()["t"], "%Y-%m-%d %H:%M:%S")
start, end, label = next_peak_window(now_local)
day_type = "Week-end" if start.weekday() >= 5 else "Semaine"
print(f"Maintenant {now_local:%a %d/%m %H:%M} → fenêtre ciblée : {label} {start:%a %d/%m %H:%M}-{end:%H:%M}")

has_forecast = spark.catalog.tableExists(f"{T}.gold_forecast_station_hourly")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Projection du stock de vélos en fin de pointe

# COMMAND ----------

forecast_cte = f"""
fc AS (
  SELECT station_id, sum(pred_net_flow) AS net_flow, count(*) AS n_hours
  FROM {T}.gold_forecast_station_hourly
  WHERE hour_local >= (SELECT h0 FROM hours_meta) AND hour_local < TIMESTAMP '{end:%Y-%m-%d %H:%M:%S}'
  GROUP BY station_id
),""" if has_forecast else "fc AS (SELECT CAST(NULL AS INT) station_id, CAST(NULL AS DOUBLE) net_flow, 0L n_hours WHERE 1 = 0),"

state = spark.sql(f"""
WITH hours_meta AS (
  SELECT date_trunc('HOUR', TIMESTAMP '{now_local:%Y-%m-%d %H:%M:%S}') + INTERVAL 1 HOUR AS h0,
         CAST((unix_timestamp(TIMESTAMP '{end:%Y-%m-%d %H:%M:%S}')
               - unix_timestamp(date_trunc('HOUR', TIMESTAMP '{now_local:%Y-%m-%d %H:%M:%S}') + INTERVAL 1 HOUR)) / 3600
              AS INT) AS n_expected
),
hours AS (
  SELECT explode(sequence(h0, TIMESTAMP '{end:%Y-%m-%d %H:%M:%S}' - INTERVAL 1 HOUR, INTERVAL 1 HOUR)) AS h
  FROM hours_meta
),
{forecast_cte}
hist AS (   -- flux net moyen historique sur les heures restantes jusqu'à la fin de la pointe
  SELECT p.station_id, sum(p.avg_net_flow) AS net_flow
  FROM hours JOIN {T}.gold_saturation_profile p
    ON p.hour_of_day = hour(hours.h)
   AND p.day_type = CASE WHEN dayofweek(hours.h) IN (1, 7) THEN 'Week-end' ELSE 'Semaine' END
  GROUP BY p.station_id
),
sat AS (    -- fréquence de saturation historique pendant la fenêtre ciblée
  SELECT station_id, avg(pct_time_empty) AS hist_pct_empty, avg(pct_time_full) AS hist_pct_full
  FROM {T}.gold_saturation_profile
  WHERE day_type = '{day_type}' AND hour_of_day >= {start.hour} AND hour_of_day < {end.hour}
  GROUP BY station_id
)
SELECT c.station_id, c.station_label, c.lat, c.lon, c.cluster_label,
       c.available_bikes, c.usable_capacity AS capacity, c.snapshot_local,
       CASE WHEN fc.n_hours >= (SELECT n_expected FROM hours_meta) THEN fc.net_flow ELSE hist.net_flow END AS expected_net_flow,
       CASE WHEN fc.n_hours >= (SELECT n_expected FROM hours_meta) THEN 'Prévision ML' ELSE 'Profil historique' END AS flow_source,
       coalesce(sat.hist_pct_empty, 0) AS hist_pct_empty,
       coalesce(sat.hist_pct_full, 0)  AS hist_pct_full
FROM {T}.gold_station_current c
LEFT JOIN fc   USING (station_id)
LEFT JOIN hist USING (station_id)
LEFT JOIN sat  USING (station_id)
WHERE c.status = 'OPEN' AND c.usable_capacity > 0
""").toPandas()

if state.empty:
    dbutils.notebook.exit("Aucune station ouverte — pas de recommandation")

state["expected_net_flow"] = state["expected_net_flow"].fillna(0.0)
state["projected_bikes"] = (state["available_bikes"] + state["expected_net_flow"]).clip(lower=0)
state["projected_bikes"] = state[["projected_bikes", "capacity"]].min(axis=1)
print(state["flow_source"].value_counts().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classement des stations et appariement

# COMMAND ----------

actions = state.apply(lambda r: classify_station(r.projected_bikes, int(r.capacity)), axis=1, result_type="expand")
state[["action", "bikes_to_move"]] = actions
state["priority"] = state.apply(
    lambda r: r.bikes_to_move * (1 + (r.hist_pct_empty if r.action == "REAPPROVISIONNER" else r.hist_pct_full)),
    axis=1)

def as_nodes(df: pd.DataFrame) -> list:
    return [{"station_id": int(r.station_id), "lat": r.lat, "lon": r.lon,
             "qty": int(r.bikes_to_move), "priority": float(r.priority)} for r in df.itertuples()]

moves = pd.DataFrame(pair_moves(
    receivers=as_nodes(state[state.action == "REAPPROVISIONNER"]),
    donors=as_nodes(state[state.action == "DELESTER"]),
    max_km=MAX_PAIR_KM,
), columns=["from_station", "to_station", "bikes", "distance_km"])

served = moves.groupby("to_station")["bikes"].sum() if not moves.empty else pd.Series(dtype=float)
state["bikes_covered_by_pairing"] = state.station_id.map(served).fillna(0).astype(int)
state.loc[state.action != "REAPPROVISIONNER", "bikes_covered_by_pairing"] = 0

names = state.set_index("station_id")["station_label"]
if not moves.empty:
    moves["from_label"] = moves.from_station.map(names)
    moves["to_label"] = moves.to_station.map(names)

summary = state.groupby("action").agg(stations=("station_id", "size"), bikes=("bikes_to_move", "sum"))
display(summary)
display(moves.head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Écriture
# MAGIC `*_latest` (remplacées à chaque exécution, pour Power BI) et `*_history` (append, pour suivre dans le temps).

# COMMAND ----------

meta = {"window_label": label, "window_start": f"{start:%Y-%m-%d %H:%M:%S}",
        "window_end": f"{end:%Y-%m-%d %H:%M:%S}", "day_type": day_type}


def to_spark(pdf: pd.DataFrame):
    """pandas → Spark ; les bornes de fenêtre passent en texte puis sont typées côté Spark."""
    return (spark.createDataFrame(pdf)
            .withColumn("window_start", F.col("window_start").cast("timestamp"))
            .withColumn("window_end", F.col("window_end").cast("timestamp"))
            .withColumn("generated_at", F.current_timestamp()))


reco = state.assign(**meta)[[
    "window_label", "window_start", "window_end", "day_type", "station_id", "station_label", "lat", "lon",
    "cluster_label", "available_bikes", "capacity", "expected_net_flow", "flow_source", "projected_bikes",
    "action", "bikes_to_move", "priority", "hist_pct_empty", "hist_pct_full", "bikes_covered_by_pairing"]]
reco_sdf = to_spark(reco)
reco_sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{T}.gold_rebalancing_latest")
reco_sdf.write.mode("append").option("mergeSchema", "true").saveAsTable(f"{T}.gold_rebalancing_history")

if not moves.empty:
    mv_sdf = to_spark(moves.assign(**meta))
    mv_sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{T}.gold_rebalancing_moves_latest")
    mv_sdf.write.mode("append").option("mergeSchema", "true").saveAsTable(f"{T}.gold_rebalancing_moves_history")
elif spark.catalog.tableExists(f"{T}.gold_rebalancing_moves_latest"):
    spark.sql(f"DELETE FROM {T}.gold_rebalancing_moves_latest")

print(f"{(state.action != 'OK').sum()} stations à traiter, {len(moves)} mouvements proposés")
