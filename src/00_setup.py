# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Setup
# MAGIC Crée le schéma Unity Catalog et les volumes utilisés par le projet.
# MAGIC À lancer **une seule fois** (idempotent : peut être relancé sans risque).
# MAGIC
# MAGIC | Objet | Rôle |
# MAGIC |---|---|
# MAGIC | `landing` (volume) | fichiers JSON bruts tels que renvoyés par les API |
# MAGIC | `checkpoints` (volume) | checkpoints Auto Loader / Structured Streaming |
# MAGIC | `bronze_*` (tables) | une ligne par fichier ingéré, payload brut conservé |

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "mobility")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema} "
          "COMMENT 'Analyse de la mobilité urbaine — vélos en libre-service + météo'")

for volume, comment in [
    ("landing", "Zone d'atterrissage des réponses API brutes (JSON)"),
    ("checkpoints", "Checkpoints Auto Loader"),
]:
    safe_comment = comment.replace("'", "''")   # échappe les apostrophes pour SQL
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume} COMMENT '{safe_comment}'")

display(spark.sql(f"SHOW VOLUMES IN {catalog}.{schema}"))
