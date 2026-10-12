# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Prévision de la fréquentation
# MAGIC Prévoit, pour chaque station et chaque heure des **24 prochaines heures**, le nombre de départs et
# MAGIC d'arrivées. Un modèle global (toutes stations) par cible :
# MAGIC
# MAGIC - **Modèle** : `HistGradientBoostingRegressor` (perte de Poisson, adaptée à des comptages) ;
# MAGIC - **Features** : heure, jour, week-end, météo (prévision Open-Meteo à l'inférence), retards J-1 et J-7,
# MAGIC   encodages par station (moyenne globale et par heure, calculés sur le train), capacité, cluster ;
# MAGIC - **Validation** : découpage temporel (7 derniers jours en test) et comparaison à une **baseline
# MAGIC   saisonnière naïve** (même heure J-7, sinon J-1). `skill = 1 - MAE_modèle / MAE_baseline`.
# MAGIC
# MAGIC Tout est suivi dans MLflow (paramètres, métriques, importance des variables, modèle). Les prévisions
# MAGIC sont écrites dans `gold_forecast_station_hourly`.

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=2.14" scikit-learn matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import inspect
import os
import sys

sys.path.insert(0, os.getcwd())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from mobility.ml_utils import add_station_encodings, regression_metrics

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "mobility")
dbutils.widgets.dropdown("register_models", "false", ["true", "false"])
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
REGISTER = dbutils.widgets.get("register_models") == "true"
T = f"{catalog}.{schema}"
spark.conf.set("spark.sql.session.timeZone", "UTC")

MIN_DAYS = 14
TEST_DAYS = 7
HISTORY_DAYS = 90
TARGETS = ["departures", "arrivals"]
FEATURES = ["hour_of_day", "weekday", "is_weekend", "temperature", "precipitation", "wind_speed",
            "lag_24h", "lag_168h", "station_mean", "station_hour_mean", "capacity", "cluster_id"]
PARAMS = {"loss": "poisson", "learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31,
          "min_samples_leaf": 40, "l2_regularization": 1.0, "early_stopping": True,
          "validation_fraction": 0.1, "random_state": 42}

# Les versions récentes de MLflow sérialisent sklearn avec skops et exigent de déclarer les types de
# confiance ; le modèle étant produit ici, on déclare l'arbre du gradient boosting (si l'option existe).
LOG_MODEL_KW = (
    {"skops_trusted_types": ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]}
    if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters else {}
)

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/mobilite_urbaine")

# COMMAND ----------

n_days = spark.sql(f"SELECT count(DISTINCT date_local) FROM {T}.gold_station_hourly").first()[0]
if n_days < MIN_DAYS:
    dbutils.notebook.exit(f"Historique insuffisant : {n_days} jour(s), minimum {MIN_DAYS}. Prévision reportée.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Jeu d'entraînement

# COMMAND ----------

has_clusters = spark.catalog.tableExists(f"{T}.gold_station_clusters")
cluster_expr = "c.cluster_id" if has_clusters else "CAST(NULL AS INT) AS cluster_id"
cluster_join = f"LEFT JOIN {T}.gold_station_clusters c ON c.station_id = h.station_id" if has_clusters else ""

# retards J-1 et J-7 de chaque cible, par auto-jointure sur l'heure
LAG_COLS = ", ".join(
    f"{a}.{t} AS {t}_lag_{lag}h" for t in TARGETS for a, lag in (("l24", 24), ("l168", 168))
)

df = spark.sql(f"""
WITH h AS (
  SELECT * FROM {T}.gold_station_hourly
  WHERE date_local >= date_sub(current_date(), {HISTORY_DAYS}) AND n_snapshots >= 2 AND pct_open > 0.5
)
SELECT h.station_id, CAST(h.hour_local AS STRING) AS hour_local, h.hour_of_day, h.weekday, h.is_weekend,
       h.temperature, h.precipitation, h.wind_speed, h.departures, h.arrivals,
       {LAG_COLS},
       d.capacity, {cluster_expr}
FROM h
LEFT JOIN h l24  ON l24.station_id  = h.station_id AND l24.hour_local  = h.hour_local - INTERVAL 24 HOURS
LEFT JOIN h l168 ON l168.station_id = h.station_id AND l168.hour_local = h.hour_local - INTERVAL 168 HOURS
LEFT JOIN {T}.silver_stations d ON d.station_id = h.station_id
{cluster_join}
WHERE h.hour_local < date_trunc('HOUR', from_utc_timestamp(current_timestamp(), 'Europe/Paris'))
""").toPandas()

df["hour_local"] = pd.to_datetime(df["hour_local"])
cutoff = df["hour_local"].max().normalize() - pd.Timedelta(days=TEST_DAYS - 1)
print(f"{len(df):,} lignes — train < {cutoff:%Y-%m-%d} ≤ test")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entraînement, évaluation, suivi MLflow

# COMMAND ----------

NUMERIC = ["hour_of_day", "weekday", "is_weekend", "temperature", "precipitation", "wind_speed",
           "lag_24h", "lag_168h", "capacity", "cluster_id"]


def prepare(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Sélectionne les retards de la cible et force des types numériques (NULL → NaN, géré nativement)."""
    out = frame.copy()
    out["lag_24h"] = out[f"{target}_lag_24h"]
    out["lag_168h"] = out[f"{target}_lag_168h"]
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    if target in out:
        out[target] = pd.to_numeric(out[target], errors="coerce").astype(float)
    return out


models, metrics_rows = {}, []
with mlflow.start_run(run_name="prevision_frequentation") as parent:
    mlflow.log_params({"history_days": n_days, "test_days": TEST_DAYS, "n_rows": len(df),
                       "features": ",".join(FEATURES), **{f"hgb_{k}": v for k, v in PARAMS.items()}})

    for target in TARGETS:
        with mlflow.start_run(run_name=f"hgb_{target}", nested=True):
            data = prepare(df, target)
            train, test = data[data.hour_local < cutoff], data[data.hour_local >= cutoff]
            train, test = add_station_encodings(train, test, target)

            model = HistGradientBoostingRegressor(**PARAMS).fit(train[FEATURES], train[target])
            pred = model.predict(test[FEATURES])

            baseline = test["lag_168h"].fillna(test["lag_24h"]).fillna(test["station_hour_mean"])
            m_model = regression_metrics(test[target], pred)
            m_base = regression_metrics(test[target], baseline)
            skill = 1 - m_model["mae"] / m_base["mae"] if m_base["mae"] else np.nan
            peak = test.is_weekend.eq(0) & test.hour_of_day.isin([7, 8, 9, 17, 18, 19])
            m_peak = regression_metrics(test.loc[peak, target], pred[peak.to_numpy()])

            mlflow.log_param("target", target)
            mlflow.log_metrics({"mae": m_model["mae"], "rmse": m_model["rmse"],
                                "baseline_mae": m_base["mae"], "skill_vs_baseline": skill,
                                "mae_peak_hours": m_peak["mae"]})

            # importance des variables (permutation, sur un échantillon du test)
            sample = test.sample(min(len(test), 20000), random_state=42)
            imp = permutation_importance(model, sample[FEATURES], sample[target],
                                         scoring="neg_mean_absolute_error", n_repeats=3, random_state=42)
            imp_df = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
                      .sort_values("importance"))
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(imp_df.feature, imp_df.importance)
            ax.set(title=f"Importance par permutation — {target}", xlabel="Hausse de MAE")
            fig.tight_layout(); mlflow.log_figure(fig, f"importance_{target}.png"); plt.close(fig)

            # réseau entier : réel vs prédit sur la période de test
            net = (test.assign(pred=pred).groupby("hour_local")[[target, "pred"]].sum())
            fig, ax = plt.subplots(figsize=(11, 3.5))
            ax.plot(net.index, net[target], label="Réel"); ax.plot(net.index, net["pred"], label="Prévu")
            ax.set(title=f"{target} — réseau, période de test"); ax.legend()
            fig.tight_layout(); mlflow.log_figure(fig, f"test_reseau_{target}.png"); plt.close(fig)

            # ré-entraînement sur tout l'historique pour la mise en production
            full, _ = add_station_encodings(data, data.head(1), target)
            final = HistGradientBoostingRegressor(**PARAMS).fit(full[FEATURES], full[target])
            mlflow.sklearn.log_model(
                final, "model",
                signature=infer_signature(full[FEATURES], final.predict(full[FEATURES].head(50))),
                input_example=full[FEATURES].head(3),
                registered_model_name=f"{catalog}.{schema}.forecast_{target}" if REGISTER else None,
                **LOG_MODEL_KW)

            models[target] = (final, data)
            metrics_rows.append({"target": target, **m_model, "baseline_mae": m_base["mae"],
                                 "skill_vs_baseline": skill, "mae_peak_hours": m_peak["mae"],
                                 "mlflow_run_id": mlflow.active_run().info.run_id})
            print(f"{target:<10} MAE {m_model['mae']:.3f} | baseline {m_base['mae']:.3f} | skill {skill:+.1%}")

    parent_run_id = parent.info.run_id

# COMMAND ----------

(spark.createDataFrame(pd.DataFrame(metrics_rows))
    .selectExpr("current_timestamp() AS trained_at", "*")
    .write.mode("append").option("mergeSchema", "true")
    .saveAsTable(f"{T}.gold_model_metrics"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prévision des 24 prochaines heures
# MAGIC Météo = prévision Open-Meteo collectée par l'ingestion ; retards J-1 / J-7 = historique observé.
# MAGIC Les heures transitent en texte entre Spark et pandas : aucune conversion de fuseau implicite.

# COMMAND ----------

future = spark.sql(f"""
WITH now AS (SELECT date_trunc('HOUR', from_utc_timestamp(current_timestamp(), 'Europe/Paris')) AS h0),
st AS (SELECT station_id, capacity FROM {T}.silver_stations
       WHERE last_seen >= current_timestamp() - INTERVAL 1 DAY),
w AS (SELECT m.time_local AS hour_local, m.temperature_2m AS temperature, m.precipitation,
             m.wind_speed_10m AS wind_speed
      FROM {T}.silver_meteo_hourly m, now
      WHERE m.time_local >= now.h0 AND m.time_local < now.h0 + INTERVAL 24 HOURS),
h AS (SELECT station_id, hour_local, departures, arrivals FROM {T}.gold_station_hourly)
SELECT st.station_id, CAST(w.hour_local AS STRING) AS hour_local, w.temperature, w.precipitation, w.wind_speed,
       hour(w.hour_local) AS hour_of_day,
       ((dayofweek(w.hour_local) + 5) % 7) + 1 AS weekday,
       CASE WHEN dayofweek(w.hour_local) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend,
       {LAG_COLS},
       st.capacity, {cluster_expr}
FROM st CROSS JOIN w
LEFT JOIN h l24  ON l24.station_id  = st.station_id AND l24.hour_local  = w.hour_local - INTERVAL 24 HOURS
LEFT JOIN h l168 ON l168.station_id = st.station_id AND l168.hour_local = w.hour_local - INTERVAL 168 HOURS
{cluster_join.replace('h.station_id', 'st.station_id')}
""").toPandas()

if future.empty:
    raise RuntimeError("Pas de prévision météo pour les 24 prochaines heures — vérifier l'ingestion météo")

out = future[["station_id", "hour_local"]].copy()
for target, (model, history) in models.items():
    _, fut = add_station_encodings(history, prepare(future, target), target)
    out[f"pred_{target}"] = np.clip(model.predict(fut[FEATURES]), 0, None)
out["pred_net_flow"] = out["pred_arrivals"] - out["pred_departures"]
out["hour_local"] = pd.to_datetime(out["hour_local"]).dt.strftime("%Y-%m-%d %H:%M:%S")
out["mlflow_run_id"] = parent_run_id

(spark.createDataFrame(out)
    .selectExpr("CAST(station_id AS INT) station_id", "CAST(hour_local AS TIMESTAMP) hour_local",
                "pred_departures", "pred_arrivals", "pred_net_flow", "mlflow_run_id",
                "current_timestamp() AS generated_at")
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{T}.gold_forecast_station_hourly"))

print(f"{len(out):,} prévisions écrites ({out.station_id.nunique()} stations × {out.hour_local.nunique()} h)")
