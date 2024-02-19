# Projet Netfloox


* **Trello** : 
https://trello.com/b/c96tOI4O/buzzing-flash-netfloox

* **La Présentation** : [La Présentation en pdf](ProjetNetfloox.pdf)

* **Documentation** : https://developer.imdb.com/non-commercial-datasets/   et   https://help.imdb.com/article/contribution/other-submission-guides/country-codes/G99K4LFRMSC37DCN# 

* **Codes SQL** : 
   * SQL.ipynb permet de créer toutes les clés primaires et de mettre les bons types dans la base de données. Il contient également les SQL pour la créations des tables et des views.

* La **Base de Donnée Relationnelle**, réalisée sous _DBeaver_ et hébergée sur _Azure_. 

  ![plot](https://github.com/coall132/Netfloox_Groupe1/blob/main/Base%20de%20Donn%C3%A9e%20Rel.PNG)https://github.com/coall132/Netfloox_Groupe1/blob/main/Base%20de%20Donn%C3%A9e%20Rel.PNG)

* **Code Python** :
  * *prep_donne.py* nous permet avec l'input d'un film de ressortir 5 films les plus similaires. Il s'appuie sur le CSV cosine_features.csv qui contient l'ensemble des films et des features importantes.
  * *Graph.py* est le script pour la réalisation des graphiques sous python.
  * On utilise le fichier *algo-reco.py* pour la recommandation d'un film.
  
 
Ce code est une mise en œuvre d'un pipeline complet pour la construction et l'évaluation de modèles de régression utilisant différents algorithmes et techniques de prétraitement des données. Les différentes bibliothèques utilisées dans le code sont importées, y compris des outils de machine learning de scikit-learn, ainsi que pandas, numpy et psycopg2 pour la manipulation des données.
Les données sont lues à partir d'un fichier CSV situé à l'URL spécifiée. En cas d'erreur de lecture, un message d'erreur est affiché. Les fonctions de nettoyage sont définies pour transformer les données textuelles dans certaines colonnes en listes, remplacer les valeurs manquantes par 'inconnu', et convertir certaines colonnes en types numériques. Ensuite, les données textuelles sont fusionnées en une seule chaîne pour chaque ligne. Un pipeline est créé  qui effectue le prétraitement des données, y compris l'imputation, la mise à l'échelle et l'encodage, puis applique un modèle d'apprentissage automatique donné. Différents ensembles de paramètres sont définis pour les différents modèles d'apprentissage automatique à évaluer. Chaque ensemble de paramètres spécifie les prétraitements et les hyperparamètres spécifiques au modèle. Les données sont divisées en ensembles d'entraînement et de test pour l'évaluation des modèles.

Un modèle final est entraîné en utilisant le meilleur pipeline identifié par la recherche sur la grille avec les meilleurs hyperparamètres. Le modèle final est évalué sur l'ensemble de test à l'aide de différentes métriques telles que le coefficient de détermination (R²), l'erreur absolue moyenne (MAE) et l'erreur quadratique moyenne (MSE). Le modèle final est sauvegardé sous forme de fichier pickle pour une utilisation ultérieure.

    * On utilise le fichier *Pop_predict.py* pour la prédiction de la notation d'un futur film.

Ce script démontre une approche complète pour la construction, l'optimisation et l'évaluation de modèles d'apprentissage automatique pour prédire des notes de films en utilisant diverses techniques de prétraitement des données et des algorithmes d'apprentissage automatique.

* **Streamlit_reco.py** contient le code pour l'application.
  Notre script combine un système de recommandation de films basé sur le contenu avec un modèle de prédiction de popularité des films. Les utilisateurs peuvent interagir avec l'application en saisissant le nom d'un film, en explorant les détails des films et en obtenant des recommandations personnalisées. Ils peuvent également utiliser le modèle pour prédire la popularité potentielle d'un film en fonction de ses caractéristiques.

  




  
