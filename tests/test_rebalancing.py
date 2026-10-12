from datetime import datetime

import pytest

from mobility.rebalancing import (classify_station, haversine_km, next_peak_window, pair_moves,
                                  thresholds)


def test_haversine_known_distance():
    # Capitole → gare Matabiau (Toulouse) ≈ 1,2 km
    d = haversine_km(43.6045, 1.4440, 43.6111, 1.4536)
    assert 1.0 < d < 1.4


def test_thresholds_use_max_of_absolute_and_relative_margin():
    assert thresholds(10) == (2, 8)      # 15 % de 10 = 1,5 → 2
    assert thresholds(40) == (6, 34)     # 15 % de 40 = 6


@pytest.mark.parametrize("projected, capacity, expected", [
    (1, 20, ("REAPPROVISIONNER", 9)),
    (19, 20, ("DELESTER", 9)),
    (10, 20, ("OK", 0)),
    (-5, 20, ("REAPPROVISIONNER", 10)),  # projection négative ramenée à 0
    (30, 20, ("DELESTER", 10)),          # projection > capacité ramenée à la capacité
    (5, 0, ("OK", 0)),
])
def test_classify_station(projected, capacity, expected):
    assert classify_station(projected, capacity) == expected


def test_next_peak_window_weekday_morning():
    start, end, label = next_peak_window(datetime(2026, 10, 12, 5, 30))   # lundi
    assert (start, end, label) == (datetime(2026, 10, 12, 7), datetime(2026, 10, 12, 10), "Pointe du matin")


def test_next_peak_window_during_morning_returns_evening():
    start, _, label = next_peak_window(datetime(2026, 10, 12, 8, 0))
    assert start == datetime(2026, 10, 12, 17) and label == "Pointe du soir"


def test_next_peak_window_friday_night_returns_saturday():
    start, end, label = next_peak_window(datetime(2026, 10, 16, 21, 0))   # vendredi
    assert start == datetime(2026, 10, 17, 14) and end.hour == 19 and label == "Après-midi week-end"


def node(sid, lat, lon, qty, priority=None):
    return {"station_id": sid, "lat": lat, "lon": lon, "qty": qty, "priority": priority or qty}


def test_pair_moves_uses_nearest_donor_first_and_respects_stock():
    receivers = [node(1, 43.600, 1.440, 6)]
    donors = [node(10, 43.601, 1.441, 4), node(11, 43.610, 1.450, 10)]
    moves = pair_moves(receivers, donors)
    assert [(m["from_station"], m["bikes"]) for m in moves] == [(10, 4), (11, 2)]


def test_pair_moves_ignores_donors_beyond_radius():
    moves = pair_moves([node(1, 43.60, 1.44, 5)], [node(10, 43.70, 1.44, 5)], max_km=3)
    assert moves == []


def test_pair_moves_serves_highest_priority_first():
    receivers = [node(1, 43.600, 1.440, 3, priority=1), node(2, 43.600, 1.441, 3, priority=9)]
    moves = pair_moves(receivers, [node(10, 43.600, 1.4405, 3)])
    assert moves[0]["to_station"] == 2 and sum(m["bikes"] for m in moves) == 3
