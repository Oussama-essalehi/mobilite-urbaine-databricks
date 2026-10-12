"""Utilitaires ML purs (pandas) : nommage des clusters, encodages par station, métriques."""
import numpy as np
import pandas as pd


def name_cluster(early_fill: float, midday_fill: float, pct_empty: float, pct_full: float,
                 delta: float = 0.15, saturation: float = 0.25) -> str:
    """Libellé métier d'un cluster à partir de son profil moyen en semaine.

    early_fill  : remplissage moyen 5h-7h (avant les départs du matin)
    midday_fill : remplissage moyen 10h-16h
    """
    if pct_empty >= saturation:
        return "Souvent vide"
    if pct_full >= saturation:
        return "Souvent pleine"
    if early_fill - midday_fill >= delta:
        return "Résidentielle (se vide le matin)"
    if midday_fill - early_fill >= delta:
        return "Pôle d'activité (se remplit le matin)"
    return "Équilibrée / loisirs"


def dedupe_labels(labels: dict) -> dict:
    """Rend les libellés uniques : deux clusters « Équilibrée » → « Équilibrée (1) », « (2) »."""
    counts, seen, out = {}, {}, {}
    for v in labels.values():
        counts[v] = counts.get(v, 0) + 1
    for k, v in sorted(labels.items()):
        if counts[v] > 1:
            seen[v] = seen.get(v, 0) + 1
            out[k] = f"{v} ({seen[v]})"
        else:
            out[k] = v
    return out


def add_station_encodings(train: pd.DataFrame, other: pd.DataFrame, target: str) -> tuple:
    """Encodage par la moyenne, calculé **sur le train uniquement** (pas de fuite) :
    - station_mean       : moyenne de la cible par station
    - station_hour_mean  : moyenne par station × type de jour × heure
    Les combinaisons inconnues retombent sur la moyenne station puis la moyenne globale.
    """
    global_mean = train[target].mean()
    st = train.groupby("station_id")[target].mean().rename("station_mean")
    sth = (
        train.groupby(["station_id", "is_weekend", "hour_of_day"])[target]
        .mean()
        .rename("station_hour_mean")
    )

    def apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.drop(columns=["station_mean", "station_hour_mean"], errors="ignore")
        out = out.join(st, on="station_id").join(sth, on=["station_id", "is_weekend", "hour_of_day"])
        out["station_mean"] = out["station_mean"].fillna(global_mean)
        out["station_hour_mean"] = out["station_hour_mean"].fillna(out["station_mean"])
        return out

    return apply(train), apply(other)


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}
