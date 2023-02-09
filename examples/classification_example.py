import mlimputer as mli
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console

#source_dataset="https://www.kaggle.com/datasets/cnic92/200-financial-indicators-of-us-stocks-20142018?select=2018_Financial_Data.csv"

## Join all years
url="https://raw.githubusercontent.com/TsLu1s/MLimputer/main/data/2018_Financial_Data.csv"

data=pd.read_csv(url, encoding='latin', delimiter=',')

# replace all instances of "[", "]" or "<" with "_" in all columns of the dataframe
data.columns = [col.replace("[", "_").replace("]", "_").replace("<", "_") for col in data.columns]

target="Class"
data=data[data[target].isnull()==False]
data=data.reset_index(drop=True)

train,test = train_test_split(data, train_size=0.8)
train,test = train.reset_index(drop=True), test.reset_index(drop=True)

# All model imputation options ->  "RandomForest","ExtraTrees","GBR","KNN","GeneralizedLR","XGBoost","Lightgbm","Catboost"

# Imputation Example 1 : RandomForest

imputer_rf=mli.fit_imput(Dataset=train,imput_model="RandomForest")
train_rf=mli.transform_imput(Dataset=train,fit_configs=imputer_rf)
test_rf=mli.transform_imput(Dataset=test,fit_configs=imputer_rf)

# Imputation Example 2 : XGBoost

imputer_xgb=mli.fit_imput(Dataset=train,imput_model="XGBoost")
train_xgb=mli.transform_imput(Dataset=train,fit_configs=imputer_xgb)
test_xgb=mli.transform_imput(Dataset=test,fit_configs=imputer_xgb)
    
## Performance Evaluation Example - Imputation CrossValidation

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import atlantic as atl

## Preprocessing Data (Label Encoder)

le_fit=atl.fit_Label_Encoding(train_xgb,target)
df=atl.transform_Label_Encoding(train_xgb,le_fit)
df=df.reset_index(drop=True)
df[target]=df[target].astype('category')

leaderboard_xgb_imp=mli.cross_validation(Dataset=df,
                                         target=target, 
                                         test_size=0.2,
                                         n_splits=3,
                                         models=[XGBClassifier(), RandomForestClassifier(), DecisionTreeClassifier()])







