# Analyse de la mobilité urbaine sur Databricks

Pipeline de données de bout en bout sur **Databricks** autour des vélos en libre-service **VélôToulouse** et de la
météo : ingestion planifiée, architecture médaillon sur **Delta Lake** avec contrôles qualité intégrés,
segmentation des stations et prévision de la fréquentation suivies dans **MLflow**, tableau de bord **Power BI**
connecté en direct au SQL warehouse, et recommandations de rééquilibrage aux heures de pointe.

**Stack :** Databricks (Free Edition, serverless) · PySpark · Spark SQL · Delta Lake · Auto Loader · Unity Catalog ·
MLflow · scikit-learn · Power BI (DirectQuery, DAX) · Databricks Asset Bundles · GitHub Actions

---

## Architecture

```
                         Job 1 — toutes les 15 min                      Job 2 — toutes les heures
GBFS VélôToulouse ─┐   ┌──────────────────────────────────────┐   ┌────────────────────────────────────┐
(temps réel)       ├──►│ 01 ingestion → landing (JSON brut)    │   │ 04 gold : faits horaires, profils,  │
API Open-Meteo ────┘   │ 02 Auto Loader → bronze (Delta)       │──►│    saturation, effet météo          │──► Power BI
                       │ 03 silver : parsing, typage, MERGE,   │   │ 07 recommandations de rééquilibrage │   (DirectQuery
                       │    expectations, quarantaine, fraîcheur│   └────────────────────────────────────┘    SQL warehouse)
                       └──────────────────────────────────────┘   Job 3 — chaque nuit (4h30)
                                                                    05 clustering des stations ─┐ MLflow
                                                                    06 prévision 24 h ──────────┘ (expériences, modèles)
```

### Couches de données (`workspace.mobility`)

| Couche | Tables | Contenu |
|---|---|---|
| **landing** | volume `landing/` | une réponse API brute par fichier JSON, partitionnée par date |
| **bronze** | `bronze_velo_raw`, `bronze_meteo_raw` | une ligne par fichier, payload conservé en texte + métadonnées d'ingestion |
| **silver** | `silver_station_status` | état de chaque station à chaque collecte (typé, dédoublonné, contrôlé) |
| | `silver_stations` | référentiel des stations (SCD 1) |
| | `silver_meteo_hourly` | météo horaire (observations et prévisions) |
| | `silver_velo_quarantine`, `dq_log` | lignes rejetées avec les règles en échec, historique des contrôles |
| **gold** | `gold_station_hourly` | **fait principal** : station × heure — départs, arrivées, remplissage, % temps vide / pleine, météo |
| | `gold_network_hourly`, `gold_dim_station`, `gold_station_current` | vue réseau, dimension station, disponibilité temps réel |
| | `gold_saturation_profile`, `gold_saturation_ranking` | profils horaires et classement des stations saturées aux heures de pointe |
| | `gold_weather_effect` | départs relatifs par condition météo et température, à heure comparable |
| **ML** | `gold_station_clusters`, `gold_forecast_station_hourly`, `gold_model_metrics` | segments, prévisions 24 h, métriques des modèles |
| **Décision** | `gold_rebalancing_latest` / `_history`, `gold_rebalancing_moves_latest` / `_history` | stations à réapprovisionner / délester, mouvements proposés |

---

## Choix techniques

**Ingestion robuste.** Reprises avec backoff exponentiel, échec explicite si une source est vide, alerte e-mail
sur échec. Le bronze stocke le JSON **brut en texte** : un changement de schéma de l'API ne casse jamais
l'ingestion, il se traduit par des lignes en quarantaine en silver.

**Incrémental et idempotent.** Auto Loader (bronze) et Structured Streaming `availableNow` + `foreachBatch`
(silver) ne traitent que les nouveautés grâce à leurs checkpoints ; les écritures silver sont des `MERGE`,
donc rejouer un fichier ne crée pas de doublon.

**Qualité intégrée au pipeline.** Règles déclarées comme des expectations SQL (`src/03_silver.py`) :
identifiant présent, comptages positifs, `vélos + bornes ≤ capacité`, coordonnées dans la métropole, statut
connu, météo plausible. Les lignes invalides vont en quarantaine ; au-delà de 20 % d'un lot ou si la dernière
collecte a plus d'une heure, la tâche échoue. En gold : unicité du grain, flux non négatifs, couverture météo.

**Estimation des flux.** L'API ne publie que des états de stations, pas des trajets. Départs et arrivées sont
estimés à partir des variations du nombre de vélos entre deux collectes successives (≤ 30 min d'écart). C'est
un minorant (un départ et une arrivée dans le même intervalle s'annulent) qui inclut aussi les opérations de
rééquilibrage par camion.

**Segmentation.** Chaque station est décrite par son profil horaire de remplissage (24 h semaine + 24 h
week-end) et sa fréquence de saturation. KMeans pour k = 3…8, choix par silhouette, chaque essai suivi dans
MLflow (graphiques de sélection, profils, ACP). Les clusters reçoivent un libellé métier :
*résidentielle (se vide le matin)*, *pôle d'activité (se remplit le matin)*, *souvent vide*, *souvent pleine*,
*équilibrée*.

**Prévision.** Un modèle global par cible (départs, arrivées) : `HistGradientBoostingRegressor` en perte de
Poisson. Variables : calendrier, météo, retards J-1 et J-7, encodages par station calculés **sur le train
uniquement**, capacité, cluster. Validation temporelle (7 derniers jours) contre une baseline saisonnière
naïve ; MAE, RMSE, MAE aux heures de pointe, gain vs baseline et importance par permutation sont journalisés
dans MLflow. À l'inférence, la météo provient de la prévision Open-Meteo.

**Recommandations.** Avant la prochaine pointe, projection du stock de chaque station en fin de pointe
(prévision ML, ou profil historique si la prévision ne couvre pas la fenêtre), seuils bas/haut
(2 vélos ou 15 % de la capacité), quantité pour revenir à 50 %, puis appariement glouton des stations à
délester vers les stations à réapprovisionner les plus proches (≤ 3 km), par priorité.

---

## Installation

### 1. Workspace
1. Créer un compte [Databricks Free Edition](https://www.databricks.com/learn/free-edition).
2. **Vérifier son identité via LinkedIn** dans le workspace : sans cela, l'accès Internet sortant du compute
   serverless est limité à quelques domaines et les appels aux API échouent.

### 2. Déployer
**Option A — Databricks Asset Bundle (recommandé)**
```bash
databricks auth login --host https://<votre-workspace>.cloud.databricks.com
# renseigner le host dans databricks.yml, puis :
databricks bundle validate
databricks bundle deploy -t dev
```
Crée les trois jobs planifiés avec leurs dépendances, reprises et alertes.

**Option B — Git folder** : *Workspace → Create → Git folder* → URL de ce dépôt, puis créer les jobs à la main
en reprenant `databricks.yml` (tâches, dépendances, planification).

### 3. Initialiser et vérifier
1. Exécuter `src/00_setup.py` (schéma + volumes).
2. Exécuter à la main `01_ingest_api` → `02_bronze` → `03_silver` → `04_gold`.
3. **Vérifier le parsing vélo** : le parsing gère les flux GBFS v3 et v2. Si `silver_velo_quarantine` se remplit, regarder `_failed_rules` et un `raw_json` de
   `bronze_velo_raw`, puis ajuster `parse_velo` dans `03_silver.py`.
4. Activer les jobs (ils le sont par défaut avec le bundle).

### 4. Calendrier de montée en charge
Les données temps réel ne s'accumulent qu'à partir du lancement : rien ne peut être récupéré rétroactivement.

| Historique | Disponible |
|---|---|
| dès la 1re heure | bronze, silver, gold, disponibilité temps réel, Power BI |
| 7 jours | clustering des stations |
| 14 jours | prévision (le notebook se met en attente proprement avant) |
| 3-4 semaines | prévision fiable (retards J-7 disponibles), profils de saturation stables |

### 5. Power BI
Voir [`powerbi/GUIDE_POWERBI.md`](powerbi/GUIDE_POWERBI.md) (connexion DirectQuery, modèle, pages) et
[`powerbi/mesures.dax`](powerbi/mesures.dax) (mesures DAX).

### Quotas Free Edition
Le job d'ingestion tourne 96 fois par jour. Si le quota d'usage quotidien est atteint, passer la planification
à 30 min dans `databricks.yml`.

---

## Développement

```bash
pip install -r requirements-dev.txt
pytest -q          # tests unitaires : rééquilibrage, fenêtres de pointe, encodages, libellés de clusters
ruff check src tests
```
La CI GitHub Actions lance le lint, les tests et la validation YAML du bundle à chaque push.

## Structure du dépôt

```
├── databricks.yml              # bundle : 3 jobs planifiés, paramètres, reprises, alertes
├── src/
│   ├── 00_setup.py             # schéma + volumes Unity Catalog
│   ├── 01_ingest_api.py        # API → landing (reprises, validation)
│   ├── 02_bronze.py            # Auto Loader → bronze
│   ├── 03_silver.py            # parsing, expectations, quarantaine, MERGE, fraîcheur
│   ├── 04_gold.py              # tables métier + contrôles gold
│   ├── 05_ml_clustering.py     # segmentation des stations (KMeans, MLflow)
│   ├── 06_ml_forecast.py       # prévision départs / arrivées 24 h (MLflow)
│   ├── 07_recommendations.py   # rééquilibrage aux heures de pointe
│   └── mobility/               # code partagé testé (qualité, rééquilibrage, utilitaires ML)
├── powerbi/                    # guide du rapport + mesures DAX
├── tests/                      # tests unitaires pytest
└── .github/workflows/ci.yml
```

## Résultats

*À compléter après 3-4 semaines de collecte* : nombre de stations et de collectes, segments obtenus,
MAE et gain du modèle vs baseline, effet de la pluie sur les départs, stations les plus saturées,
captures du rapport Power BI.

## Sources et licences

| Source | Licence |
|---|---|
| [VélôToulouse — flux GBFS (JCDecaux / Cyclocity)](https://mobilitydatabase.org/feeds/gbfs/gbfs-toulouse), découverte : `https://api.cyclocity.fr/contracts/toulouse/gbfs/v3/gbfs.json` | flux public, sans clé d'API |
| [Open-Meteo](https://open-meteo.com/) | CC BY 4.0 |
