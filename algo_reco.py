# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:52:17 2024

@author: kaeli
"""

#%%Partie pour streamlit
# Importing necessary libraries
import csv  # For handling CSV files
import pandas as pd  # For handling DataFrames
import numpy as np  # For numerical computations
from sklearn.metrics.pairwise import cosine_similarity  # For cosine similarity calculation
from sklearn.feature_extraction.text import CountVectorizer  # For converting text data into numerical format
cv=CountVectorizer()
# Reading the DataFrame from CSV file
url = "C:/Users/kaeli/Documents/python/df_recommandation_film.csv"
df = pd.read_csv(url, encoding='utf-8', on_bad_lines='warn')
df_reco_film=df.drop(['tconst','averagerating','numvotes'],axis=1)
df_reco_film=df_reco_film.drop_duplicates()

# Filtering DataFrame for films and series
df_film = df_reco_film[df_reco_film.titletype.str.contains('1')]
df_serie = df_reco_film[df_reco_film.titletype.str.contains('3')]

# Loading the saved count matrices
count_matrix_film = cv.fit_transform(df_film['features'])
count_matrix_serie = cv.fit_transform(df_serie['features'])

# Creating dictionaries to map title types to corresponding count matrices and DataFrames
c_ = {'1': count_matrix_film,
      '3': count_matrix_serie}
d_ = {'1': df_film,
      '3': df_serie}
#%%
# Function to find the closest films to a given film
def film_le_plus_proche(a, df=df_reco_film, n=5):
    index = df[df['originaltitle'].str.contains(a)].index[0]  # Finding the index of the given film on the df global
    b = df.iloc[index]['titletype']  # Finding the title type of the given film
    
    c = c_.get(b)  # Getting the corresponding count matrix ( film/serie)
    df1 = d_.get(b)  # Getting the corresponding DataFrame (film/serie)
    index1 = df1[df1['originaltitle'].str.contains(a)].index[0]  # Finding the index of the given film in its DataFrame 
    cos = cosine_similarity(c, c[index1])  # Calculating cosine similarity with all films for one line
    
    cos1 = []  # Flattened list to store cosine similarities ( transform a 2Darray in a 1D array)
    for sublist in cos:
        cos1.extend(sublist)
    sort = np.argsort(cos1)  # Sorting the cosine similarities
    liste_plus_proche = sort[::-1]  # Reversing the order to get the closest films first
    liste_final = []  # List to store the closest films
    
    for i in range(1, (n + 1)):  # Iterating over the closest films
        plus_proche = liste_plus_proche[i]  # Finding the index of the i-th closest film
        liste_final.append(df1.iloc[plus_proche, 8])  # Adding the title of the i-th closest film to the list
        
    return liste_final  # Returning the list of closest films
#%%
print(film_le_plus_proche('Titanic'))
#%%
liste2 = df_reco_film[df_reco_film['originaltitle'].str.contains('Party Monster')]