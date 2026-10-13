<h1 align="center">🚲 VéloScope Toulouse</h1>

<p align="center">
  <b>Les vélos en libre-service de Toulouse, collectés, analysés et prévus en temps réel.</b>
</p>

<p align="center">
  <a href="https://toulouse-mobilite.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Ouvrir l'application">
  </a>
  <br>
  <a href="https://toulouse-mobilite.streamlit.app/"><b>toulouse-mobilite.streamlit.app</b></a>
</p>

---

## Le projet

Une station est **vide** quand on veut prendre un vélo, **pleine** quand on veut le rendre. Ces déséquilibres
suivent les rythmes de la ville (trajets domicile-travail, week-ends, météo) : on peut donc les mesurer et les
anticiper.

VéloScope collecte en continu l'état des **~450 stations VélôToulouse** et la météo, puis les transforme en
réponses concrètes :

- 📊 **Analyser** : heures de pointe, effet de la pluie et du froid, stations souvent vides ou pleines
- 🧩 **Segmenter** : familles de stations selon leur comportement (résidentielle, pôle d'activité…)
- 🔮 **Prévoir** : départs et arrivées attendus sur les 24 prochaines heures
- 🚚 **Recommander** : quels vélos déplacer, et où, avant la prochaine pointe

## Architecture

<table align="center">
  <tr>
    <th align="center">① Sources</th>
    <th></th>
    <th align="center">② Collecte<br><sub>toutes les 15 min</sub></th>
    <th></th>
    <th align="center">③ Lakehouse<br><sub>toutes les heures</sub></th>
    <th></th>
    <th align="center">④ Machine learning<br><sub>chaque nuit</sub></th>
    <th></th>
    <th align="center">⑤ Restitution</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://img.icons8.com/color/96/bicycle.png" width="40" alt="VélôToulouse"><br>
      <b>VélôToulouse</b><br><sub>flux GBFS</sub>
      <br><br>
      <img src="https://github.com/open-meteo.png?size=96" width="40" alt="Open-Meteo"><br>
      <b>Open-Meteo</b><br><sub>météo</sub>
    </td>
    <td align="center">➜</td>
    <td align="center">
      <img src="https://cdn.simpleicons.org/databricks" width="40" alt="Databricks"><br>
      <b>Databricks Jobs</b><br><sub>serverless</sub>
      <br><br>
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="40" alt="Python"><br>
      <b>Python</b><br><sub>appels API</sub>
    </td>
    <td align="center">➜</td>
    <td align="center">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/apachespark/apachespark-original.svg" width="40" alt="Apache Spark"><br>
      <b>PySpark · Spark SQL</b>
      <br><br>
      <img src="https://github.com/delta-io.png?size=96" width="40" alt="Delta Lake"><br>
      <b>Delta Lake</b><br><sub>Unity Catalog</sub>
      <br><br>
      🥉 Bronze <sub>brut</sub><br>
      ⬇<br>
      🥈 Silver <sub>nettoyé, contrôlé</sub><br>
      ⬇<br>
      🥇 Gold <sub>tables d'analyse</sub>
    </td>
    <td align="center">➜</td>
    <td align="center">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="40" alt="scikit-learn"><br>
      <b>scikit-learn</b><br><sub>KMeans<br>gradient boosting</sub>
      <br><br>
      <img src="https://cdn.simpleicons.org/mlflow" width="40" alt="MLflow"><br>
      <b>MLflow</b><br><sub>suivi des modèles</sub>
    </td>
    <td align="center">➜</td>
    <td align="center">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/streamlit/streamlit-original.svg" width="40" alt="Streamlit"><br>
      <b>Streamlit</b><br><sub>application web</sub>
      <br><br>
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/plotly/plotly-original.svg" width="40" alt="Plotly"><br>
      <b>Plotly</b><br><sub>cartes, graphiques</sub>
      <br><br>
      <img src="https://img.icons8.com/color/96/power-bi.png" width="40" alt="Power BI"><br>
      <b>Power BI</b><br><sub>DirectQuery, DAX</sub>
    </td>
  </tr>
  <tr>
    <td colspan="9" align="center">
      <b>Industrialisation</b>&nbsp;&nbsp;
      <img src="https://cdn.simpleicons.org/databricks" width="20" alt="Databricks"> Asset Bundles (3 jobs planifiés)&nbsp;&nbsp;·&nbsp;&nbsp;
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/githubactions/githubactions-original.svg" width="20" alt="GitHub Actions"> GitHub Actions (CI)&nbsp;&nbsp;·&nbsp;&nbsp;
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytest/pytest-original.svg" width="20" alt="pytest"> pytest · ruff
    </td>
  </tr>
</table>

## Points techniques

- **Trajets déduits, pas observés** : les API ne publient que le stock de vélos par station ; les départs et
  arrivées sont reconstruits à partir des variations entre deux relevés.
- **Qualité des données** : les lignes incohérentes partent en quarantaine, et une alerte est levée si le taux de
  rejet est trop élevé ou si les données ne sont plus à jour.
- **Prévisions validées** : le modèle de gradient boosting est comparé à une référence simple (« même heure la
  semaine dernière ») pour prouver qu'il apporte un gain réel.
- **Rééquilibrage** : avant chaque pointe, le stock est projeté et les stations en manque sont associées aux
  stations excédentaires à moins de 3 km.

## L'application

Huit pages en français, lisibles sans connaissances techniques : **carte des stations en direct**, activité heure
par heure, effet de la météo, stations à problème, familles de stations, prévisions 24 h, déplacements de vélos
recommandés et suivi du pipeline.

## Sources de données

- [VélôToulouse, flux GBFS](https://mobilitydatabase.org/feeds/gbfs/gbfs-toulouse) (JCDecaux / Cyclocity) : état des stations en temps réel
- [Open-Meteo](https://open-meteo.com/) : météo observée et prévue (CC BY 4.0)

---

<p align="center">Projet réalisé par <b>Oussama Essalehi</b></p>
