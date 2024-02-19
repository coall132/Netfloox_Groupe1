# -*- coding: utf-8 -*-
"""
Laurence Berville
17.02.2024

@author: Mpadmin
"""
import os

# Obtenir le répertoire courant du projet
current_directory = os.getcwd()

print("Le répertoire courant du projet est :", current_directory)

# Nouveau chemin vers le répertoire
new_directory = "E:/Documents/4- Projets/3- Projet Netfloox/3- Projet Netfloox "

# Changer le répertoire courant
os.chdir(new_directory)

# Vérifier si le répertoire a bien été changé
current_directory = os.getcwd()
print("Le nouveau répertoire courant est :", current_directory)


#%% Packages
import pandas as pd
from plotnine import ggplot, aes, geom_bar, ggtitle, xlab, ylab, scale_fill_brewer, guides
from plotnine import options, scales, theme_minimal, scale_fill_manual
from plotnine.themes import theme
from plotnine import element_text,geom_point, geom_segment, theme_classic, scale_color_manual
import matplotlib.pyplot as plt
import numpy as np

import pkg_resources
#%% Connaitre les packages et les exporter dans le fichier requierments

# Récupérer une liste de tous les packages installés
installed_packages = pkg_resources.working_set

# Créer un dictionnaire pour stocker les noms des packages et leurs versions
packages_info = {}

# Parcourir chaque package installé et stocker son nom et sa version dans le dictionnaire
for package in installed_packages:
    packages_info[package.key] = package.version

# Afficher les noms des packages et leurs versions
for package, version in packages_info.items():
    print(f"{package}: {version}")


#%% Tables  

# Spécifiez le chemin du fichier CSV
csv_file_path = "category_job.csv"

# Importez le fichier CSV en utilisant pandas
# Assurez-vous de spécifier l'encodage UTF-8 et le délimiteur '\t' pour les tabulations
dataJob = pd.read_csv(csv_file_path, encoding='utf-8', delimiter='\t')


csv_file_path = "nbr_language.csv"
dataLang = pd.read_csv(csv_file_path, encoding='utf-8', delimiter='\t')

# Spécifiez le chemin du fichier CSV
csv_file_path = "nbr_region.csv"
dataReg = pd.read_csv(csv_file_path, encoding='utf-8', delimiter='\t')



#%%  JOB
options.figure_size = (8, 6)  # Réglez la taille du graphique

# Trier la colonne "Metier" par ordre alphabétique
data_sorted = dataJob.sort_values(by='Metier')

# Créez un graphique en utilisant ggplot
# Définissez les données (data_sorted), l'esthétique (aes) et le type de graphique (geom_bar)
# Utilisez également ggtitle, xlab et ylab pour définir le titre et les étiquettes des axes
p = (
    ggplot(data_sorted, aes(x='Metier', y='Nombre', fill='Metier')) +  # Définissez les données et l'esthétique
    geom_bar(stat='identity') +  # Utilisez un graphique en barres
    ggtitle("Nombre de personnes par catégorie de métier") +  # Définissez le titre du graphique
    xlab("Métier") +  # Définissez l'étiquette de l'axe des abscisses
    ylab("Nombre en milliers") +  # Définissez l'étiquette de l'axe des ordonnées
    scale_fill_brewer(type='qual', palette='Set3') +  # Définissez la palette de couleurs
    theme_minimal() +  # Utilisez un thème minimal pour le graphique
    scales.scale_x_discrete(labels=lambda l: sorted(l)) + # Trier la colonne "Metier" par ordre alphabétique
    guides(fill=False) +  # Enlever la légende
    theme(axis_text_x=element_text(angle=90, hjust=1))  # Faites pivoter le texte de l'axe x
)

# Mettre les chiffres de l'axe y en milliers
p = p + scales.scale_y_continuous(labels=lambda l: ["{:,.0f}".format(v / 1000) for v in l])

# Afficher le graphique dans la console de Spyder
print(p)


#%% Langues-------------------------------------------------------------------------------------------------------

### Camembert
# Calculer le nombre total de films
total_films = dataLang['Nombre'].sum()

# Calculer le pourcentage de chaque langue
dataLang['Pourcentage'] = (dataLang['Nombre'] / total_films) * 100

# Trier les données par le nombre de films (dans l'ordre décroissant)
data2_sorted = dataLang.sort_values(by='Nombre', ascending=False)

# Filtrer les données pour ne garder que les langues dont le pourcentage est supérieur ou égal à 0.1%
data2_filtered = data2_sorted[data2_sorted['Pourcentage'] >= 0.1]

# Liste de couleurs pour chaque langue
colors = plt.cm.tab10.colors[:len(data2_filtered)]

# Exploser le pie chart pour mettre en évidence certaines tranches
explode = [0.05] * len(data2_filtered)  # Réglez le niveau d'éloignement pour chaque tranche

# Créer un pie chart avec une couleur différente pour chaque langue et des tranches exploser
plt.figure(figsize=(8, 8))
plt.pie(data2_filtered['Pourcentage'], labels=data2_filtered['Language'], 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors, 
        explode=explode)
plt.title("Répartition en pourcentage des langues des films (> 0.1% )")
plt.axis('equal')  # Assurez-vous que le pie chart est parfaitement circulaire
plt.show()

# ----------------------------------------------------------------------------------------------


# Trier les données par pourcentage de films par langue de manière décroissante
top_20_languages =  dataLang.sort_values(by='Pourcentage', ascending=False).head(10)

# Créer un graphique en utilisant ggplot
p = (
    ggplot(top_20_languages, aes(x='Language', y='Pourcentage', fill='Language')) +  
    geom_bar(stat='identity') +  
    ggtitle("Top 10 : Genre de ""vidéos"" en pourcentage par langue") +  
    xlab("Langues") +  
    ylab("%") +  
    theme_minimal() +
    scale_fill_manual(values=['#440154', '#472777', '#3E4A89', '#31688E', '#26838E', '#1F9E89', '#35B779', '#6DCC71', '#B4DE2C', '#FDE725'])
)

# Afficher le graphique
print(p)

#%% Regions---------------------------------------------------------------------------------------------------

data3_sorted = dataReg.sort_values(by='Nombre', ascending=False)
# Sélectionner les 40 premières entrées après le tri par ordre alphabétique
top_40_origins = data3_sorted.head(40)

# Créer une palette de couleurs avec 40 couleurs différentes
palette = [
   '#FF0000', '#FF4500', '#FF7F00', '#FFA500', '#FFD700',
    '#FFFF00', '#FFEC80', '#FFFF33', '#C1FF33', '#80FF00',
    '#00FF00', '#33FF33', '#66FF33', '#99FF33', '#CCFF33',
    '#997300', '#664200', '#332100', '#000000', '#333333',
    '#666666', '#999999', '#B2B2B2', '#CCCCCC', '#E5E5E5',
    '#FFFFFF', '#E5FFFF', '#CCFFFF', '#B2FFFF', '#99FFFF',
    '#80FFFF', '#00FFFF', '#33CCFF', '#3399FF', '#3366FF',
    '#3333FF', '#3300FF', '#4B0082', '#6600CC', '#9900CC',
    '#9900FF', '#B266FF', '#CC33FF', '#FF00FF', '#FF33FF'
]

# Créer un graphique en utilisant ggplot avec une palette de couleurs personnalisée
p = (
    ggplot(top_40_origins, aes(x='Nombre', y='Origine', color='Origine')) +  
    geom_point(size=3) +  # Utiliser des points pour représenter les données
    geom_segment(aes(x=0, xend='Nombre', y='Origine', yend='Origine'), 
                 color='blue') +  # Utiliser des segments pour les "lollipops"
    theme_classic() +
    scale_color_manual(values=palette)
)

# Afficher le graphique
print(p)


#%% Notes ------------------------------------------------------------------------------------------------

# Spécifiez le chemin du fichier TSV
tsv_file_path = "title_ratings.tsv"

# Lire le fichier TSV en utilisant pandas
data4 = pd.read_csv(tsv_file_path, encoding='utf-8', delimiter='\t')

# Définir les couleurs en fonction des valeurs de averageRating
colors = np.where(data4['averageRating'] >= 8, 'green',   # Vert pour averageRating >= 8
                  np.where(data4['averageRating'] >= 6, 'orange',  # Orange pour 6 <= averageRating < 8
                           'red'))   # Rouge pour averageRating < 6

# Créer un graphique de dispersion avec des couleurs basées sur averageRating
plt.scatter(data4['averageRating'], data4['numVotes'], c=colors)
plt.title('Corrélation entre averageRating et numVotes')
plt.xlabel('averageRating')
plt.ylabel('numVotes')
plt.show()

#%% Correlations----------------------------------------------------------------------------------------------------
# Import des bibliothèques
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Spécifiez le chemin du fichier CSV
csv_file_path = "analysesinteret.csv"

# Lire le fichier CSV en utilisant pandas
dataInteret = pd.read_csv(csv_file_path, encoding='utf-8')

# Utilisez l'argument 'hue' pour spécifier une variable de facteur
sns.lmplot(x="averagerating", y="startyear", data=dataInteret, 
           fit_reg=True, order=1, hue='numvotes')

# Basic correlogram
sns.pairplot(dataInteret)
plt.show()
 
# with regression
sns.pairplot(dataInteret, kind="reg") # long et inutile
plt.show()
 
# without regression
sns.pairplot(dataInteret, kind="scatter")
plt.show()

# %% barre plot genre

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Spécifiez le chemin du fichier CSV
csv_file_path = "genre_populaireuniquenbr.csv"

# Lire le fichier CSV en utilisant pandas
dataGenres = pd.read_csv(csv_file_path, encoding='utf-8')

# Plot the histogram using the distplot function
plt.figure(figsize=(10, 6))  # Définir la taille de la figure

# Définir une palette de couleurs pour chaque genre
palette = sns.color_palette("hls", len(dataGenres['genre']))

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='genre', y='nbr', data=dataGenres, palette=palette)
plt.xlabel('Genres')
plt.ylabel('Nombre de Votes')
plt.title('Nombre de Votes par Genre')
plt.xticks(rotation=90)
plt.show()
# Afficher le graphique
plt.show()

#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Spécifiez le chemin du fichier CSV
csv_file_path = "stattemps.csv"

# Lire le fichier CSV en utilisant pandas
dataTemps = pd.read_csv(csv_file_path, encoding='utf-8')


# Créer un graphique de corrélation
plt.figure(figsize=(8, 6))  # Définir la taille de la figure

# Tracer un nuage de points
plt.scatter(dataTemps['runtimeminutes'], dataTemps['numvotes'], alpha=0.5)

# Étiqueter les axes
plt.xlabel('Durée du film (minutes)')
plt.ylabel('Nombre de votes')
plt.title('Corrélation entre la durée du film et le nombre de votes')

# Afficher le graphique
plt.show()


#%%  Date de films - notation
import pandas as pd
from plotnine import ggplot, aes, geom_boxplot

# Spécifiez le chemin du fichier CSV
csv_file_path = "statyrating.csv"

# Lire le fichier CSV en utilisant pandas
dataYearRat = pd.read_csv(csv_file_path, encoding='utf-8')

# Créer le graphique en utilisant plotnine
plot = (ggplot(dataYearRat) +
        aes(x='factor(startyear)', y='averagerating') +
        geom_boxplot())

# Afficher le graphique
print(plot)

import pandas as pd
from plotnine import ggplot, aes, geom_point

# Spécifiez le chemin du fichier CSV
csv_file_path = "statyrating.csv"

# Lire le fichier CSV en utilisant pandas
dataYearRat = pd.read_csv(csv_file_path, encoding='utf-8')

# Traitement des valeurs non définies
dataYearRat['startyear'].fillna(0, inplace=True)  # Remplacer les valeurs NaN par 0 ou une autre valeur par défaut


# Créer le graphique en utilisant plotnine
plot = (ggplot(dataYearRat) +
        aes(x='staryear', y='averagerating') +
        geom_point() )

# Afficher le graphique
print(plot)


