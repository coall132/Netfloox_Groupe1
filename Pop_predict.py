# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:49:14 2024

@author: kaeli
"""
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import psycopg2
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder

from dotenv import load_dotenv
import os

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

url="C:/Users/kaeli/Documents/python/df_recommandation_film.csv"
try:
    df = pd.read_csv(url, encoding='utf-8',on_bad_lines='warn')
except pd.errors.ParserError as e:
    print("Error parsing CSV file:", e)
    
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
df=df.replace(np.nan,'inconnus')
#%%
df['runtimeminutes']= df['runtimeminutes'].replace('inconnus',np.nan)
df['numvotes']= df['numvotes'].replace('inconnus',np.nan)
df['averagerating']= df['averagerating'].replace('inconnus',np.nan)
df['startyear']= df['startyear'].replace('inconnus',np.nan)
df.dropna(subset=['averagerating'], inplace=True)
df=df.drop(['tconst','language'],axis=1)

#%%

def pipeline(model=DecisionTreeRegressor(), num=[SimpleImputer(),StandardScaler()], text=[OneHotEncoder(handle_unknown='ignore')]):
    colonne_numerique=['startyear','runtimeminutes','numvotes']
    colonne_nominal=[colonne for colonne in X.columns if colonne not in colonne_numerique]
    
    pip_num = Pipeline(steps=[(f"etape_num_{count}", i) for count, i in enumerate(num)])
    pip_text = Pipeline(steps=[(f"etape_text_{count}", i) for count, i in enumerate(text)])
    
    preprocess = ColumnTransformer(transformers=[('text', pip_text, colonne_nominal),
                                                 ('num', pip_num, colonne_numerique)])
    model = Pipeline(steps=[('preprocessor', preprocess), ('model', model)])
    return model


#steps=[(f"etape_num_{count}",i) for count,i in enumerate(num)]
#pip_num=Pipeline(steps=[f"""('etape{count}',{i}){"," if i!=num[-1] else ''}""" for count,i in enumerate(num)])


def hGS(df,param,cv):
    y = df['averagerating']
    X = df.drop(['averagerating','titletype','features'],axis=1) #df['message']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
  
    multi_scoring= ['r2','MAE','MSE']
    
    grid = HalvingGridSearchCV(estimator=pipeline(),param_grid=param,scoring='neg_mean_squared_error',cv=cv,n_jobs=-1, verbose=1, error_score="raise")
    
    grid.fit(X_train, y_train)
    
    best_score = grid.best_score_
    best_params = grid.best_params_
    training_time = grid.cv_results_['mean_fit_time'].mean()
    return({'best_score': best_score,
            'best_params': best_params,
            'training_time': training_time,
            'fitted_model': grid.best_estimator_})



param = [{
    'preprocessor__num__etape_num_0':[SimpleImputer(strategy='mean')],
    'preprocessor__num__etape_num_1':[StandardScaler()],
    'preprocessor__text__etape_text_0':[OneHotEncoder(handle_unknown='ignore')],
    'model': [RandomForestRegressor()],
    'model__criterion':['squared_error','absolute_error'],
    'model__max_depth':[30],
    'model__n_estimators':[20,50,100,150]}]
param1 =[  {
     'preprocessor__num__etape_num_0':[SimpleImputer(strategy='mean')],
     'preprocessor__num__etape_num_1':[StandardScaler()],
     'preprocessor__text__etape_text_0':[OneHotEncoder(handle_unknown='ignore')],
     'model': [AdaBoostRegressor()],
     'model__n_estimators': [100,500,1000],
     'model__loss': ['linear', 'square', 'exponential']}]
param2 = [   {
     'preprocessor__num__etape_num_0':[SimpleImputer(strategy='mean')],
     'preprocessor__num__etape_num_1':[StandardScaler()],
     'preprocessor__text__etape_text_0':[OneHotEncoder(handle_unknown='ignore')],
     'model': [GradientBoostingRegressor()],
     'model__loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
     'model__n_estimators': [100,500,1000],
     'model__criterion': ['friedman_mse', 'squared_error'],
     'model__max_depth': [3,5,10]}]

#%%
def join_features(row):
    features = []
    if isinstance(row, str):
        return row
    for item in row:
        if isinstance(item, list):
            features.extend(item)
        else:
            features.append(item)
    merged_features = [item.replace(' ', '') if isinstance(item, str) else str(item) for item in features]
    return ' '.join(map(str, merged_features))

df['directors']=df['directors'].apply(join_features)
df['writers']=df['writers'].apply(join_features)
df['genres_type']=df['genres_type'].apply(join_features)
df['actor']=df['actor'].apply(join_features)
'''
def concatenate_columns_as_string(row):
    return ' '.join(map(str, row))

df['features'] = df[['directors','writers','genres_type','actor','originaltitle']].apply(concatenate_columns_as_string, axis=1)
'''
#%%

st=RobustScaler()
y = df['averagerating']
X =df.drop(['averagerating','titletype','features'],axis=1) #df['message']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#%%
a=pipeline(model=RandomForestRegressor(n_estimators=100,max_depth=30,criterion='squared_error',n_jobs=-1),num=[SimpleImputer(strategy='mean'),st],text=[OneHotEncoder(handle_unknown='ignore')])
model_final=a.fit(X_train, y_train)

y_pred = a.predict(X_test)
print(r2_score(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

#%%
import pickle

with open('modele_pred_pop.pkl', 'wb') as model_file:
    pickle.dump(model_final, model_file)

#%%
reponse=hGS(df,param,cv=5)
