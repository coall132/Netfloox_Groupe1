# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 14:19:32 2024

@author: kaeli
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle

cv=CountVectorizer()

if st.button('load the data'):
    url = "df_recommandation_film.csv"
    if 'df_reco_film' not in st.session_state:
        st.session_state.df_reco_film = pd.read_csv(url, encoding='utf-8', on_bad_lines='warn')
    if 'count_matrix_film' not in st.session_state:
        st.session_state.count_matrix_film = cv.fit_transform(st.session_state.df_reco_film['features'])
    if 'model' not in st.session_state:
        with open('modele_pred_pop.pkl', 'rb') as model_file:
            st.session_state.model = pickle.load(model_file)

        
def film_le_plus_proche(a, df,c, n=5):  # Finding the index of the given film on the df global
    b = df.iloc[a]['titletype']# Finding the title type of the given film
    
    ligne_film = df.iloc[a,:]
    
    cos = cosine_similarity(c,c[a])  # Calculating cosine similarity with all films for one line
    cos1 = []  # Flattened list to store cosine similarities ( transform a 2Darray in a 1D array)
    cos1=[item for liste in cos for item in liste]
    sort = np.argsort(cos1)  # Sorting the cosine similarities
    liste_plus_proche = sort[::-1]  # Reversing the order to get the closest films first
    
    liste_final = []  # List to store the closest films
    for i in range(1, (n + 1)):  # Iterating over the closest films
        plus_proche = liste_plus_proche[i]  # Finding the index of the i-th closest film
        liste_final.append(plus_proche)
    return liste_final   # Returning the list of closest films        

        
st.title('Popcorn Movies')

st.session_state.movie_name = st.text_input('Enter the name of a movie:')

if st.button('Choose a name'):
    if not st.session_state.movie_name:
        st.write('Please enter a movie name.')
    else:
        exemple = st.session_state.df_reco_film[st.session_state.df_reco_film['originaltitle'].str.contains(st.session_state.movie_name)]
        st.dataframe(exemple[['originaltitle','startyear','actor','writers','directors','genres_type','numvotes']], width=800)
st.session_state.index_film = st.number_input('Enter the index of the movie:')
st.session_state.index_film=int(st.session_state.index_film)

if st.button('get recomandation') :
    if not st.session_state.index_film:
        st.write('Please enter an index.')
    else:
        
        st.session_state.liste_plus_proche_fin= film_le_plus_proche(st.session_state.index_film,st.session_state.df_reco_film,st.session_state.count_matrix_film)
        st.session_state.film_final = st.session_state.df_reco_film.iloc[[i for i in st.session_state.liste_plus_proche_fin], :]
        st.write(f"le film choisi est : {st.session_state.df_reco_film.iloc[st.session_state.index_film]['originaltitle']}")
        st.dataframe(st.session_state.film_final[['originaltitle','startyear','actor','writers','directors','genres_type','numvotes']], width=800)
        
st.title('Prédiction de la popularité')
    
st.subheader("Entrez les données (ne pas mettre d'espace entre les nom, prénom d'une personne'):")

directors = st.text_input('directors')
writers = st.text_input('writers')
genres_type = st.text_input('genres')
actor = st.text_input('actor')
originaltitle = st.text_input('title')
    
startyear = st.number_input('startyear')
runtimeminutes = st.number_input('duration of the movie')
numvotes = st.number_input('number of votes')

startyear=int(startyear)
runtimeminutes=int(runtimeminutes)
numvotes=int(numvotes)

if st.button('Prédire'):
    data = {
        'directors': directors,
        'writers': writers,
        'genres_type': genres_type,
        'originaltitle': originaltitle,
        'actor': actor,
        'startyear': startyear,
        'runtimeminutes': runtimeminutes,
        'numvotes': numvotes
        }   
    df_predict = pd.DataFrame(data,index=[0])
    
    prediction = st.session_state.model.predict(df_predict)
    df_predict = pd.concat([pd.DataFrame(prediction, columns=['prediction']), df_predict], axis=1)
    st.dataframe(df_predict)