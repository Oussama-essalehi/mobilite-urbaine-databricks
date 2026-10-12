# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Landing → Bronze (Auto Loader)
# MAGIC Charge de façon **incrémentale** les nouveaux fichiers JSON de la zone landing dans deux tables Delta :
# MAGIC `bronze_velo_raw` et `bronze_meteo_raw`.
# MAGIC
# MAGIC - Auto Loader (`cloudFiles`) mémorise les fichiers déjà lus dans son checkpoint → pas de doublon, pas de fichier oublié.
# MAGIC - `trigger(availableNow=True)` : traite ce qui est arrivé puis s'arrête (adapté à un job planifié et au serverless).
# MAGIC - Le payload est stocké **en texte brut** : le bronze ne casse jamais si l'API change de schéma ;
# MAGIC   le parsing et le typage se font en silver.

# COMMAND ----------

from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "mobility")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
LANDING = f"/Volumes/{catalog}/{schema}/landing"
CHECKPOINTS = f"/Volumes/{catalog}/{schema}/checkpoints"

# COMMAND ----------

def load_bronze(source: str):
    target = f"{catalog}.{schema}.bronze_{source}_raw"

    stream = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "text")
        .option("wholetext", "true")              # 1 fichier = 1 ligne
        .load(f"{LANDING}/{source}/")
        .select(
            F.col("value").alias("raw_json"),
            F.get_json_object("value", "$.fetched_at").cast("timestamp").alias("fetched_at"),
            F.col("_metadata.file_path").alias("source_file"),
            F.col("_metadata.file_modification_time").alias("file_modified_at"),
            F.current_timestamp().alias("ingested_at"),
        )
    )

    query = (
        stream.writeStream
        .option("checkpointLocation", f"{CHECKPOINTS}/bronze_{source}")
        .trigger(availableNow=True)
        .toTable(target)
    )
    query.awaitTermination()
    return target

# COMMAND ----------

for source in ["velo", "meteo"]:
    table = load_bronze(source)
    n = spark.table(table).count()
    print(f"{table} : {n} fichiers ingérés au total")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Contrôle rapide

# COMMAND ----------

display(spark.sql(f"""
    SELECT 'velo'  AS source, count(*) AS nb_fichiers, min(fetched_at) AS premier, max(fetched_at) AS dernier
    FROM {catalog}.{schema}.bronze_velo_raw
    UNION ALL
    SELECT 'meteo', count(*), min(fetched_at), max(fetched_at)
    FROM {catalog}.{schema}.bronze_meteo_raw
"""))
