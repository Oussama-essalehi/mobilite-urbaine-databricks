# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Segmentation des stations (clustering)
# MAGIC Chaque station est décrite par son **profil horaire de remplissage** (24 h en semaine + 24 h le week-end)
# MAGIC et sa fréquence de saturation sur les 8 dernières semaines. Un KMeans est entraîné pour k = 3…8 ;
# MAGIC chaque essai est suivi dans **MLflow** (silhouette, inertie) et le meilleur k est retenu.
# MAGIC Les clusters reçoivent ensuite un libellé métier (résidentielle, pôle d'activité, souvent vide…).

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=2.14" scikit-learn matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.getcwd())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mobility.ml_utils import dedupe_labels, name_cluster

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "mobility")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
T = f"{catalog}.{schema}"

MIN_DAYS = 7
LOOKBACK_DAYS = 56
K_RANGE = range(3, 9)

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/mobilite_urbaine")

# COMMAND ----------

n_days = spark.sql(f"SELECT count(DISTINCT date_local) FROM {T}.gold_station_hourly").first()[0]
if n_days < MIN_DAYS:
    dbutils.notebook.exit(f"Historique insuffisant : {n_days} jour(s), minimum {MIN_DAYS}. Clustering reporté.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Matrice de features

# COMMAND ----------

profile = spark.sql(f"""
    SELECT station_id, day_type, hour_of_day, avg(avg_fill_rate) AS fill
    FROM {T}.gold_station_hourly
    WHERE date_local >= date_sub(current_date(), {LOOKBACK_DAYS}) AND pct_open > 0.5
    GROUP BY ALL
""").toPandas()

saturation = spark.sql(f"""
    SELECT station_id, avg(pct_time_empty) AS pct_empty, avg(pct_time_full) AS pct_full
    FROM {T}.gold_station_hourly
    WHERE date_local >= date_sub(current_date(), {LOOKBACK_DAYS}) AND pct_open > 0.5
    GROUP BY station_id
""").toPandas().set_index("station_id")

X = profile.pivot_table(index="station_id", columns=["day_type", "hour_of_day"], values="fill")
X.columns = [f"{'we' if d == 'Week-end' else 'wd'}_fill_{h:02d}" for d, h in X.columns]
X = X.apply(lambda row: row.fillna(row.mean()), axis=1).dropna()   # heure manquante → moyenne de la station
X = X.join(saturation, how="inner").astype(float)
print(f"{X.shape[0]} stations × {X.shape[1]} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entraînement et sélection de k

# COMMAND ----------

results = {}
with mlflow.start_run(run_name="clustering_stations") as parent:
    mlflow.log_params({"lookback_days": LOOKBACK_DAYS, "n_stations": X.shape[0],
                       "n_features": X.shape[1], "history_days": n_days})

    for k in K_RANGE:
        with mlflow.start_run(run_name=f"kmeans_k{k}", nested=True):
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=42))])
            labels = pipe.fit_predict(X)
            sil = silhouette_score(pipe["scaler"].transform(X), labels)
            mlflow.log_param("k", k)
            mlflow.log_metrics({"silhouette": sil, "inertia": pipe["kmeans"].inertia_})
            results[k] = {"silhouette": sil, "inertia": pipe["kmeans"].inertia_, "pipe": pipe, "labels": labels}

    best_k = max(results, key=lambda k: results[k]["silhouette"])
    best = results[best_k]
    mlflow.log_params({"best_k": best_k})
    mlflow.log_metric("best_silhouette", best["silhouette"])

    # --- libellés métier
    assigned = X.assign(cluster_id=best["labels"])
    wd_cols = lambda hours: [f"wd_fill_{h:02d}" for h in hours if f"wd_fill_{h:02d}" in X.columns]
    summary = assigned.groupby("cluster_id").agg(
        n_stations=("pct_empty", "size"), pct_empty=("pct_empty", "mean"), pct_full=("pct_full", "mean"))
    summary["early_fill"] = assigned.groupby("cluster_id")[wd_cols(range(5, 8))].mean().mean(axis=1)
    summary["midday_fill"] = assigned.groupby("cluster_id")[wd_cols(range(10, 17))].mean().mean(axis=1)
    raw_labels = {cid: name_cluster(r.early_fill, r.midday_fill, r.pct_empty, r.pct_full)
                  for cid, r in summary.iterrows()}
    summary["cluster_label"] = summary.index.map(dedupe_labels(raw_labels))
    mlflow.log_table(summary.reset_index(), "cluster_summary.json")

    # --- graphiques
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ks = list(results)
    ax[0].plot(ks, [results[k]["silhouette"] for k in ks], marker="o")
    ax[0].set(title="Silhouette selon k", xlabel="k")
    ax[1].plot(ks, [results[k]["inertia"] for k in ks], marker="o", color="tab:orange")
    ax[1].set(title="Inertie (coude)", xlabel="k")
    fig.tight_layout(); mlflow.log_figure(fig, "selection_k.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    hours = [int(c[-2:]) for c in X.columns if c.startswith("wd_fill_")]
    for cid, grp in assigned.groupby("cluster_id"):
        ax.plot(hours, grp[[c for c in X.columns if c.startswith("wd_fill_")]].mean(),
                label=f"{summary.loc[cid, 'cluster_label']} (n={len(grp)})")
    ax.set(title="Remplissage moyen en semaine par cluster", xlabel="Heure", ylabel="Taux de remplissage")
    ax.legend(fontsize=8); fig.tight_layout(); mlflow.log_figure(fig, "profils_clusters.png"); plt.close(fig)

    coords = PCA(n_components=2, random_state=42).fit_transform(best["pipe"]["scaler"].transform(X))
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=best["labels"], cmap="tab10", s=18)
    ax.set(title=f"Stations projetées (ACP) — k={best_k}")
    fig.tight_layout(); mlflow.log_figure(fig, "acp_clusters.png"); plt.close(fig)

    mlflow.sklearn.log_model(best["pipe"], "model",
                             signature=infer_signature(X, best["labels"]), input_example=X.head(3))
    run_id = parent.info.run_id

print(f"Meilleur k = {best_k} (silhouette {best['silhouette']:.3f})")
display(summary.reset_index())

# COMMAND ----------

out = assigned[["cluster_id"]].reset_index()
out["cluster_label"] = out["cluster_id"].map(summary["cluster_label"])
out["k"], out["silhouette"], out["mlflow_run_id"] = best_k, float(best["silhouette"]), run_id

(spark.createDataFrame(out)
    .selectExpr("CAST(station_id AS INT) station_id", "CAST(cluster_id AS INT) cluster_id", "cluster_label",
                "CAST(k AS INT) k", "silhouette", "mlflow_run_id", "current_timestamp() AS trained_at")
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{T}.gold_station_clusters"))
print(f"{T}.gold_station_clusters mis à jour ({len(out)} stations)")
