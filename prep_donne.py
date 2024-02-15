# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:37:49 2024

@author: kaeli
"""

import psycopg2
import csv
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from dotenv import load_dotenv
import os

schema='public'

load_dotenv()
url="C:/Users/kaeli/Documents/python/view_fin.csv"
try:
    df = pd.read_csv(url, encoding='utf-8',on_bad_lines='warn',nrows=435000)
except pd.errors.ParserError as e:
    print("Error parsing CSV file:", e)
#%%
df=df.replace('nan',np.nan)
df=df.replace([''],np.nan)

def tolst(a):
    if a is not np.NAN:
        a=a.replace('{','').replace('}','')
        a=a.split(",")
    if a==['']:
        a=np.nan
    return a
df['directors']=df['directors'].apply(tolst)
df['writers']=df['writers'].apply(tolst)
df['genres_type']=df['genres_type'].apply(tolst)
df['language']=df['language'].apply(tolst)
df['actor']=df['actor'].apply(tolst)
#%%
df=df.fillna("")
#%%

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer()

df['features'] = df.apply(lambda row: ' '.join(map(str, row)), axis=1)
df.loc[:,'titletype'] = df['titletype'].replace({'movie':'1','short':'1','tvShort':'1','tvMovie':'1','video':'1','tvEpisode':'2','tvSeries':'3','tvMiniSeries':'3'})

df_film=df[df.titletype.str.contains('1')]
df_episode=df[df.titletype.str.contains('2')]
df_serie=df[df.titletype.str.contains('3')]

count_matrix_film = cv.fit_transform(df_film['features'])
count_matrix_episode = cv.fit_transform(df_episode['features'])
count_matrix_serie = cv.fit_transform(df_serie['features'])
cosine_sim_film = cosine_similarity(count_matrix_film)
cosine_sim_episode = cosine_similarity(count_matrix_episode)
cosine_sim_serie = cosine_similarity(count_matrix_serie)

c_map = {
    '1': cosine_sim_film,
    '2': cosine_sim_episode,
    '3': cosine_sim_serie}
d_map = {
    '1': df_film,
    '2': df_episode,
    '3': df_serie}

def film_le_plus_proche(df,n):
    a=input('choisis un film')
    index = df[df['primarytitle'].str.contains(a)].index[0]
    b = df.iloc[index,'titletype']
    c = c_map.get(b)
    df1= d_map.get(b)
    index1= df1[df1['primarytitle'].str.contains(a)].index[0]
    sort = np.argsort(c[index1])
    liste_plus_proche = sort[::-1]
    print(f"Les 5 films les plus proches de {df1.iloc[index1,2]} sont: ")
    for i in range(1,(n+1)):
        plus_proche = liste_plus_proche[i+1]
        print(f"  -{i}/{df1.iloc[plus_proche,2]}")

film_le_plus_proche(df,5)
