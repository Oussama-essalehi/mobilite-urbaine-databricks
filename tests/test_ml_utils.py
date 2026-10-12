import pandas as pd
import pytest

from mobility.ml_utils import add_station_encodings, dedupe_labels, name_cluster, regression_metrics


@pytest.mark.parametrize("args, expected", [
    ((0.8, 0.3, 0.05, 0.05), "Résidentielle (se vide le matin)"),
    ((0.2, 0.7, 0.05, 0.05), "Pôle d'activité (se remplit le matin)"),
    ((0.5, 0.5, 0.05, 0.05), "Équilibrée / loisirs"),
    ((0.8, 0.3, 0.40, 0.05), "Souvent vide"),
    ((0.5, 0.5, 0.05, 0.30), "Souvent pleine"),
])
def test_name_cluster(args, expected):
    assert name_cluster(*args) == expected


def test_dedupe_labels():
    out = dedupe_labels({0: "A", 1: "B", 2: "A"})
    assert out == {0: "A (1)", 1: "B", 2: "A (2)"}


def test_station_encodings_computed_on_train_only():
    train = pd.DataFrame({"station_id": [1, 1, 2], "is_weekend": [0, 0, 0],
                          "hour_of_day": [8, 8, 8], "y": [2.0, 4.0, 10.0]})
    test = pd.DataFrame({"station_id": [1, 3], "is_weekend": [0, 0], "hour_of_day": [9, 8], "y": [100.0, 100.0]})
    tr, te = add_station_encodings(train, test, "y")
    assert tr.loc[0, "station_hour_mean"] == 3.0
    # station connue, heure inconnue → moyenne de la station ; station inconnue → moyenne globale du train
    assert te.loc[0, "station_hour_mean"] == 3.0
    assert te.loc[1, "station_mean"] == pytest.approx(16 / 3)


def test_regression_metrics():
    m = regression_metrics([0, 2, 4], [1, 2, 1])
    assert m["mae"] == pytest.approx(4 / 3)
    assert m["rmse"] == pytest.approx((10 / 3) ** 0.5)
