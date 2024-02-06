
import psycopg2
import csv
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from dotenv import load_dotenv
import os

load_dotenv()
BDD_URL = os.environ['url_conn']
print(BDD_URL)
engine = create_engine(BDD_URL)
#%% import fichier 
def charge_csv(fn):
    fn2=fn.replace('_','.')
    url = f"https://datasets.imdbws.com/{fn2}.tsv.gz"
    a=pd.read_csv(url, compression='gzip',sep='\t',encoding='utf-8',chunksize=50000,nrows=100000)
    #df = pd.concat(a, ignore_index=True)
    #df=df.replace(r'\\N', np.NaN, regex=True)
    #if fn=='title_akas':
        #df=df.rename(columns={'titleId':'tconst'})
    #df.to_sql(f'{fn}1', con=engine, schema=schema, if_exists='replace', index=False,chunksize=50000)
    for df in a:
        print(df.shape)

#%%creations des tables
connection = psycopg2.connect(
    user=os.environ['user'],
    password=os.environ['password'],
    host=os.environ['host'],
    port=int(os.environ['port']),
    database=os.environ['database'],
    sslmode=os.environ['sslmode'],
    options=os.environ['options']
)
cursor = connection.cursor()
schema='kaelig'

TABLES= {}
TABLES['title_basics']=(f"CREATE TABLE {schema}.title_basics ("
    "tconst varchar(11),"
    "titleType TEXT,"
    "primaryTitle TEXT,"
    "originalTitle TEXT,"
    "isAdult bool,"
    "startYear int,"
    "endYear int,"
    "runtimeMinutes int,"
    "genres TEXT,"
    "PRIMARY KEY (tconst)"
    ")")

TABLES['title_crew']=(f"CREATE TABLE IF NOT EXISTS {schema}.title_crew ("
    "tconst varchar(11),"
    "directors TEXT,"
    "writers TEXT,"
    "PRIMARY KEY (tconst)"
    ")")

TABLES['title_episode']=(f"CREATE TABLE IF NOT EXISTS {schema}.title_episode ("
    "tconst varchar(11),"
    "parentTconst varchar(11),"
    "seasonNumber int,"
    "episodeNumber int,"
    "PRIMARY KEY (tconst)"
    ")")

TABLES['title_principal']=(f"CREATE TABLE IF NOT EXISTS {schema}.title_principal ("
    "tconst varchar(11),"
    "ordering int,"
    "nconst varchar(11),"
    "category TEXT,"
    "job TEXT,"
    "characters TEXT,"
    "PRIMARY KEY (tconst,ordering)"
    ")")

TABLES['title_ratings']=(f"CREATE TABLE IF NOT EXISTS {schema}.title_ratings ("
    "tconst varchar(11),"
    "averageRating DECIMAL(30,9),"
    "numVotes int,"
    "PRIMARY KEY (tconst)"
    ")")

TABLES['name_basics']=(f"CREATE TABLE IF NOT EXISTS {schema}.name_basics ("
    "nconst varchar(11),"
    "primaryName TEXT,"
    "birthYear int,"
    "deathYear int,"
    "primaryProfession TEXT,"
    "knownForTitles TEXT,"
    "PRIMARY KEY (nconst)"
    ")")

TABLES['title_akas']=(f"CREATE TABLE IF NOT EXISTS {schema}.title_akas ("
    "tconst varchar(11),"
    "ordering int,"
    "title TEXT,"
    "region TEXT,"
    "language TEXT,"
    "types TEXT,"
    "attributes TEXT,"
    "isOriginalTitle bool,"
    "genres TEXT,"
    "PRIMARY KEY (tconst, ordering)"
    ")")

for key in TABLES:
    print(f'{key} begin')
    cursor.execute(f'DROP TABLE IF EXISTS {schema}.{key}')
    cursor.execute(TABLES[key])
    charge_csv(key)
    print(f'{key} end')
