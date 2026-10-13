# VéloScope Toulouse

**Les vélos en libre-service de Toulouse, collectés, analysés et prévus en temps réel.**

VéloScope observe en continu les quelque 450 stations VélôToulouse et la météo, puis transforme ces données en
réponses simples : quand et où les Toulousains prennent le vélo, ce que change la pluie, quelles stations manquent
de vélos ou de places, et où déplacer des vélos avant les heures de pointe.

---

## Le problème

Tous les usagers d'un service de vélos en libre-service connaissent la situation : la station est **vide** quand
on veut prendre un vélo, ou **pleine** quand on veut le rendre. Ces déséquilibres ne sont pas dus au hasard : ils
suivent les rythmes de la ville (trajets domicile-travail, week-ends, météo). Si on les mesure, on peut les
anticiper.

## Ce que fait le projet

1. **Collecter** — toutes les 15 minutes, sans intervention humaine, le nombre de vélos et de places libres dans
   chaque station, ainsi que la météo.
2. **Vérifier et ranger** — les données incohérentes (station inconnue, plus de vélos que de places…) sont écartées
   automatiquement et mises de côté pour analyse.
3. **Analyser** — heures de pointe, effet de la pluie et de la température, stations souvent vides ou souvent
   pleines.
4. **Prévoir et conseiller** — estimer l'affluence des 24 prochaines heures et proposer des déplacements de vélos
   entre stations avant les pointes.

## L'application

Le résultat est présenté dans un site web en français, pensé pour être lu sans connaissances techniques.

| Page | Ce qu'on y voit |
|---|---|
| **Les vélos en ce moment** | la carte de toutes les stations avec leur état en direct (vide, pleine, disponible) |
| **Quand roule-t-on ?** | l'activité heure par heure, en semaine et le week-end |
| **L'effet de la météo** | combien de trajets en moins quand il pleut ou qu'il fait froid |
| **Stations à problème** | les stations le plus souvent vides ou pleines aux heures de pointe |
| **Familles de stations** | les stations regroupées selon leur comportement : quartiers résidentiels qui se vident le matin, pôles d'activité qui se remplissent… |
| **Prévisions 24 heures** | les départs et arrivées attendus, comparés à ce qui s'est réellement passé |
| **Déplacer les vélos** | les stations à réapprovisionner ou à délester avant la prochaine pointe, et les trajets de camion proposés |
| **Comment ça marche** | le fonctionnement du pipeline et le suivi de la qualité des données |

L'application fonctionne de deux façons : branchée sur Databricks, elle affiche les **données réelles** mises à jour
en continu ; sans connexion, elle affiche une **démonstration** (vraies stations, historique simulé).

## Comment ça marche

```
   Sources                     Collecte (toutes les 15 min)        Analyse (toutes les heures)        Résultats
┌─────────────────┐          ┌────────────────────────────┐      ┌────────────────────────────┐
│ VélôToulouse    │          │  bronze : réponses brutes  │      │  gold : tables d'analyse   │      Site web
│ (état stations) │ ───────► │  silver : données propres  │ ───► │  recommandations           │ ───► Streamlit
│ Open-Meteo      │          │           et vérifiées     │      ├────────────────────────────┤
│ (météo)         │          └────────────────────────────┘      │  Machine learning (nuit) : │      Rapport
└─────────────────┘                                              │  familles + prévisions     │ ───► Power BI
                                                                 └────────────────────────────┘
```

Les données traversent trois étapes, selon l'**architecture « médaillon »** :

- **Bronze** — chaque réponse des API est conservée telle quelle. Si le format d'une API change, rien ne casse :
  on peut toujours retraiter l'historique.
- **Silver** — les données sont décodées, typées, dédoublonnées et contrôlées. Les lignes invalides partent en
  quarantaine ; si trop de lignes sont rejetées ou si les données ne sont plus à jour, une alerte est envoyée.
- **Gold** — les tables prêtes à l'emploi : activité par station et par heure, profils de saturation, effet de la
  météo, état actuel du réseau.

Trois tâches planifiées font tourner l'ensemble : la collecte toutes les 15 minutes, l'analyse et les
recommandations toutes les heures, et le machine learning chaque nuit.

### Quelques idées clés

- **Estimer les trajets sans les voir.** Les API publient le nombre de vélos par station, pas les trajets. Les
  départs et arrivées sont déduits des variations entre deux relevés successifs.
- **Regrouper les stations.** Chaque station est décrite par la façon dont elle se remplit au fil de la journée.
  Un algorithme de regroupement (KMeans) en tire des familles, nommées ensuite en langage courant
  (*résidentielle*, *pôle d'activité*, *souvent vide*…).
- **Prévoir l'affluence.** Un modèle de gradient boosting apprend à partir de l'heure, du jour, de la météo prévue
  et de l'activité de la veille et de la semaine précédente. Il est comparé à une méthode simple (« même heure la
  semaine dernière ») pour vérifier qu'il apporte vraiment quelque chose.
- **Proposer des déplacements.** Avant chaque pointe, le stock de chaque station est projeté. Les stations qui
  vont manquer de vélos sont associées aux stations proches (moins de 3 km) qui en auront trop.

Les analyses s'enrichissent avec le temps : les familles de stations apparaissent après une semaine de collecte,
les prévisions après deux semaines, et deviennent fiables au bout de trois à quatre semaines.

## Technologies

| Domaine | Outils |
|---|---|
| Plateforme de données | Databricks (serverless), Delta Lake, Unity Catalog, Auto Loader |
| Traitement | PySpark, Spark SQL |
| Machine learning | scikit-learn, MLflow (suivi des expériences et des modèles) |
| Visualisation | Streamlit, Plotly, Power BI (DirectQuery, DAX) |
| Industrialisation | Databricks Asset Bundles (jobs planifiés), GitHub Actions (tests et lint), pytest |

## Sources de données

| Source | Contenu | Licence |
|---|---|---|
| [VélôToulouse — flux GBFS](https://mobilitydatabase.org/feeds/gbfs/gbfs-toulouse) (JCDecaux / Cyclocity) | état des stations en temps réel | flux public, sans clé |
| [Open-Meteo](https://open-meteo.com/) | météo observée et prévue | CC BY 4.0 |

---

## Lancer le projet

### L'application seule

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Sans configuration, l'application s'ouvre en mode démonstration. Pour la brancher sur Databricks, copier
`.streamlit/secrets.toml.example` en `.streamlit/secrets.toml` et renseigner les informations du SQL warehouse.

### Le pipeline sur Databricks

1. Créer un compte [Databricks Free Edition](https://www.databricks.com/learn/free-edition) (vérifier son identité
   dans le workspace pour autoriser les appels aux API externes).
2. Renseigner l'adresse du workspace dans `databricks.yml`, puis déployer les notebooks et les trois jobs :
   ```bash
   databricks auth login --host https://<workspace>.cloud.databricks.com
   databricks bundle deploy -t dev
   ```
3. Exécuter une fois `src/00_setup.py`, puis laisser tourner les jobs.

Le rapport Power BI est décrit dans [`powerbi/GUIDE_POWERBI.md`](powerbi/GUIDE_POWERBI.md).

### Tests

```bash
pip install pandas numpy pytest ruff
pytest -q
ruff check src tests dashboard scripts
```

## Organisation du dépôt

```
├── src/                 # pipeline Databricks : ingestion, bronze, silver, gold, ML, recommandations
│   └── mobility/        # code partagé et testé (qualité, rééquilibrage, outils ML)
├── dashboard/           # application Streamlit (pages, accès aux données, données de démonstration)
├── powerbi/             # guide du rapport et mesures DAX
├── scripts/             # génération des données de démonstration
├── tests/               # tests unitaires
└── databricks.yml       # définition des trois jobs planifiés
```

---

Projet réalisé par **Oussama Essalehi**.
