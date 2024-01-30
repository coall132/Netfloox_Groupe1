# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:09:33 2024

@author: kaeli
"""
import sqlite3
import csv
import pandas as pd
import numpy as np

title_principal = pd.read_csv("C:/Users/kaeli/Documents/python/title_principal.tsv",sep='\t',nrows=1000)
title_akas = pd.read_csv("C:/Users/kaeli/Documents/python/title_akas.tsv",sep='\t',nrows=1000)
title_basics = pd.read_csv("C:/Users/kaeli/Documents/python/title_basics.tsv",sep='\t',nrows=1000)
title_crew = pd.read_csv("C:/Users/kaeli/Documents/python/title_crew.tsv",sep='\t',nrows=1000)
title_episode = pd.read_csv("C:/Users/kaeli/Documents/python/title_episode.tsv",sep='\t',nrows=1000)
title_ratings = pd.read_csv("C:/Users/kaeli/Documents/python/title_ratings.tsv",sep='\t',nrows=1000)
name_basics = pd.read_csv("C:/Users/kaeli/Documents/python/name_basics.tsv",sep='\t',nrows=1000)

title_principal=title_principal.replace(r'\\N', np.nan, regex=True)
title_akas=title_akas.replace(r'\\N', np.nan, regex=True)
title_basics=title_basics.replace(r'\\N', np.nan, regex=True)
title_crew=title_crew.replace(r'\\N', np.nan, regex=True)
title_episode=title_episode.replace(r'\\N', np.nan, regex=True)
title_ratings=title_ratings.replace(r'\\N', np.nan, regex=True)
name_basics=name_basics.replace(r'\\N', np.nan, regex=True)

con = sqlite3.connect("C:/Users/kaeli/sqlite/Database")
cur = con.cursor()
#cur.execute("CREATE TABLE movie(title TEXT, year TEXT, score INTEGER)")



