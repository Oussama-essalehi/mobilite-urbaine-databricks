"""Logique de rééquilibrage : fenêtres de pointe, seuils, classement des stations, appariement.

Fonctions pures (sans Spark) pour être testées unitairement.
"""
import math
from datetime import datetime, timedelta

# (jour_type, heure_début, heure_fin, libellé) — heures locales Europe/Paris
PEAK_WINDOWS = [
    ("Semaine", 7, 10, "Pointe du matin"),
    ("Semaine", 17, 20, "Pointe du soir"),
    ("Week-end", 14, 19, "Après-midi week-end"),
]


def next_peak_window(now: datetime) -> tuple:
    """Prochaine fenêtre de pointe qui *commence* après `now` (on rééquilibre avant la pointe).

    Retourne (début, fin, libellé).
    """
    for day_offset in range(0, 8):
        day = (now + timedelta(days=day_offset)).date()
        day_type = "Week-end" if day.weekday() >= 5 else "Semaine"
        for wtype, start_h, end_h, label in PEAK_WINDOWS:
            if wtype != day_type:
                continue
            start = datetime(day.year, day.month, day.day, start_h)
            if start > now:
                return start, datetime(day.year, day.month, day.day, end_h), label
    raise RuntimeError("aucune fenêtre de pointe trouvée")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def thresholds(capacity: int, min_abs: int = 2, min_pct: float = 0.15) -> tuple:
    """Seuils bas / haut : au moins `min_abs` vélos et `min_abs` bornes libres (ou 15 % de la capacité)."""
    margin = max(min_abs, math.ceil(min_pct * capacity))
    return margin, capacity - margin


def classify_station(projected_bikes: float, capacity: int, target_pct: float = 0.5) -> tuple:
    """Action recommandée et quantité de vélos à déplacer pour revenir à `target_pct` de remplissage.

    Retourne (action, quantité) avec action ∈ {REAPPROVISIONNER, DELESTER, OK}.
    """
    if capacity is None or capacity <= 0:
        return "OK", 0
    projected = min(max(projected_bikes, 0), capacity)
    low, high = thresholds(capacity)
    target = round(target_pct * capacity)
    if projected < low:
        return "REAPPROVISIONNER", int(round(target - projected))
    if projected > high:
        return "DELESTER", int(round(projected - target))
    return "OK", 0


def pair_moves(receivers: list, donors: list, max_km: float = 3.0) -> list:
    """Appariement glouton : chaque station à réapprovisionner (par priorité décroissante)
    est servie par les stations à délester les plus proches, dans un rayon de `max_km`.

    receivers / donors : listes de dict {station_id, lat, lon, qty, priority}.
    Retourne une liste de mouvements {from_station, to_station, bikes, distance_km}.
    """
    stock = {d["station_id"]: dict(d) for d in donors}
    moves = []
    for r in sorted(receivers, key=lambda x: -x.get("priority", x["qty"])):
        need = r["qty"]
        candidates = sorted(
            (
                (haversine_km(r["lat"], r["lon"], d["lat"], d["lon"]), sid)
                for sid, d in stock.items()
                if d["qty"] > 0
            ),
            key=lambda t: t[0],
        )
        for dist, sid in candidates:
            if need <= 0 or dist > max_km:
                break
            qty = min(need, stock[sid]["qty"])
            moves.append(
                {"from_station": sid, "to_station": r["station_id"], "bikes": qty,
                 "distance_km": round(dist, 2)}
            )
            stock[sid]["qty"] -= qty
            need -= qty
    return moves
