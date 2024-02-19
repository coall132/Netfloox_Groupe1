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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

schema='public'

load_dotenv()
url="C:/Users/kaeli/Documents/python/netfloox_final.csv"
try:
    df = pd.read_csv(url, encoding='utf-8',on_bad_lines='warn')
    df=df.drop_duplicates()
except pd.errors.ParserError as e:
    print("Error parsing CSV file:", e)
    


#%%
df=df.replace('nan','inconnus')
df=df.replace([''],np.nan)
#%%

def tolst(a):
    if a is not np.NAN:
        a=a.replace('{','').replace('}','').replace('[','').replace(']','').replace('"','').replace("'",'')
        a=a.split(",")
    if a==['']:
        a=np.nan
    return a
df['directors']=df['directors'].apply(tolst)
df['writers']=df['writers'].apply(tolst)
df['genres_type']=df['genres_type'].apply(tolst)
df['language']=df['language'].apply(tolst)
df['actor']=df['actor'].apply(tolst)
df['language'] = df['language'].apply(lambda liste: [x.replace(' en', 'en') for x in liste] if isinstance(liste, list) else liste)
df['language']=df['language'].apply(lambda x: list(set(x)) if isinstance(x, list) else x)
df_reco=df.replace(np.nan,'inconnus')


#%%
cv = CountVectorizer()

def join_features(row):
    features = []
    for col in row.index:
        if isinstance(row[col], list):
            features.extend(row[col])
        else:
            features.append(row[col])
    return ' '.join(map(str, features))

df_reco['features'] = df_reco.loc[:,['startyear','writers','actor','directors','runtimeminutes','averagerating','originaltitle','numvotes']].apply(join_features, axis=1)
#%%


df_reco.loc[:,'titletype'] = df_reco['titletype'].replace({'movie':'1','short':'1','tvShort':'1','tvMovie':'1','video':'1','tvEpisode':'2','tvSeries':'3','tvMiniSeries':'3'})
df_reco.to_csv('C:/Users/kaeli/Documents/python/df_recommandation_film.csv', index=False)

