# Tableau de bord Power BI — connexion directe au SQL warehouse Databricks

## 1. Connexion

1. Dans Databricks : **SQL Warehouses** → ouvrir le warehouse serverless → onglet **Connection details** :
   noter *Server hostname* et *HTTP path*.
2. Créer un jeton : **Settings → Developer → Access tokens → Generate new token**.
3. Power BI Desktop : **Obtenir les données → Azure Databricks** (le connecteur « Databricks » fonctionne aussi) :
   - Server hostname / HTTP path : ceux de l'étape 1 ;
   - Catalog : `workspace` · Database : `mobility` ;
   - Mode : **DirectQuery** ;
   - Authentification : *Personal Access Token*.
4. Sélectionner les tables listées ci-dessous.

> Le warehouse s'éteint quand il est inactif : la première requête après une pause prend quelques secondes.

## 2. Modèle de données (modèle composite)

| Table | Stockage | Rôle |
|---|---|---|
| `gold_station_hourly` | DirectQuery | fait principal |
| `gold_dim_station` | **Double** | dimension station |
| `Date` (table DAX, voir `mesures.dax`) | Import | calendrier |
| `gold_station_current` | DirectQuery | disponibilité temps réel |
| `gold_forecast_station_hourly` | DirectQuery | prévisions 24 h |
| `gold_saturation_ranking` | DirectQuery | classement saturation |
| `gold_network_hourly` | DirectQuery | vue réseau |
| `gold_weather_effect` | Import | effet météo (petite table) |
| `gold_rebalancing_latest`, `gold_rebalancing_moves_latest` | DirectQuery | recommandations |
| `dq_log`, `gold_model_metrics` | Import | qualité & modèles |

**Relations** (cardinalité plusieurs-à-un, filtrage simple) :

- `gold_dim_station[station_id]` → `gold_station_hourly`, `gold_station_current`, `gold_forecast_station_hourly`,
  `gold_saturation_ranking`, `gold_rebalancing_latest` (`station_id`)
- `Date[Date]` → `gold_station_hourly[date_local]` et `gold_network_hourly[date_local]`

Catégoriser `lat` / `lon` en *Latitude* / *Longitude* pour les cartes.

## 3. Pages du rapport

| Page | Visuels |
|---|---|
| **Fréquentation** | cartes KPI (Départs, Arrivées, Évolution vs S-1) · courbe départs par heure, semaine vs week-end · matrice heure × jour (carte de chaleur) · top 10 stations |
| **Disponibilité** | carte des stations colorée par `availability_status` · jauges Vélos disponibles / Bornes libres · nombre de stations vides et pleines · horodatage de la dernière collecte |
| **Effet de la météo** | barres `relative_to_reference` par catégorie météo et par tranche de température · nuage de points départs × température · KPI Effet pluie |
| **Stations saturées** | tableau classé (`gold_saturation_ranking`) avec mise en forme conditionnelle · carte par `main_issue` · profil horaire de la station sélectionnée |
| **Prévision & rééquilibrage** | courbe réel vs prévu (réseau) · tableau des stations à réapprovisionner / délester · tableau des mouvements proposés (de → vers, vélos, distance) · carte |
| **Qualité & modèles** | taux d'échec par règle (`dq_log`) · évolution MAE / skill (`gold_model_metrics`) |

Segments communs : Date, type de jour, cluster, station.

## 4. Bonnes pratiques DirectQuery

- Garder les visuels sur des agrégats (heure, jour, cluster) ; éviter les tableaux de milliers de lignes non filtrés.
- Activer **Réduire les requêtes** (Options → Réduction des requêtes) pour appliquer les segments en un clic.
- Les mesures sont écrites pour se traduire en SQL simple (`SUM`, `AVERAGE`, `DIVIDE`) : pas d'itérateurs
  ligne à ligne sur la table de faits.
- Pour publier sur le service Power BI : renseigner les identifiants de la source (jeton) dans les paramètres
  du modèle sémantique ; aucune passerelle n'est nécessaire pour Databricks.
