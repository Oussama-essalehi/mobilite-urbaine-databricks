"""Contrôles qualité déclaratifs (type « expectations ») pour les DataFrames Spark.

Chaque règle est une expression SQL booléenne. Une ligne qui viole au moins une règle
(ou dont l'expression vaut NULL) est mise en quarantaine avec la liste des règles en échec.
"""
from datetime import datetime, timezone

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def split_by_expectations(df: DataFrame, rules: dict) -> tuple:
    """Retourne (lignes_valides, lignes_invalides). Les invalides portent `_failed_rules`."""
    checks = [
        F.when(~F.coalesce(F.expr(expr), F.lit(False)), F.lit(name))
        for name, expr in rules.items()
    ]
    flagged = df.withColumn("_failed_rules", F.array_compact(F.array(*checks)))
    valid = flagged.filter(F.size("_failed_rules") == 0).drop("_failed_rules")
    invalid = flagged.filter(F.size("_failed_rules") > 0)
    return valid, invalid


def summarize(df: DataFrame, rules: dict) -> dict:
    """Nombre total de lignes et nombre d'échecs par règle (une seule passe sur les données)."""
    aggs = [F.count(F.lit(1)).alias("__total")] + [
        F.sum(F.when(~F.coalesce(F.expr(expr), F.lit(False)), 1).otherwise(0)).alias(name)
        for name, expr in rules.items()
    ]
    row = df.agg(*aggs).first().asDict()
    return {k: int(v or 0) for k, v in row.items()}


def log_quality(spark, log_table: str, table_name: str, counts: dict) -> None:
    """Ajoute une ligne par règle dans la table de suivi qualité (lue par Power BI)."""
    total = counts.get("__total", 0)
    if total == 0:
        return
    run_ts = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = [
        (run_ts, table_name, rule, failed, total)
        for rule, failed in counts.items()
        if rule != "__total"
    ]
    (
        spark.createDataFrame(
            rows, "run_ts timestamp, table_name string, rule string, failed_rows long, total_rows long"
        )
        .write.mode("append")
        .saveAsTable(log_table)
    )
