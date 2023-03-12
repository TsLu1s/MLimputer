import mlimputer as mli
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console

#source_data="https://www.kaggle.com/datasets/cnic92/200-financial-indicators-of-us-stocks-20142018"

url="https://raw.githubusercontent.com/TsLu1s/MLimputer/main/data/2018_Financial_Data.csv"
data= pd.read_csv(url) # Dataframe Loading Example

data.isna().sum()
data.dtypes

target="Class"
data[target]=data[target].astype('category')
data=data[data[target].isnull()==False]
data=data.reset_index(drop=True)

train,test = train_test_split(data, train_size=0.8) 
train,test = train.reset_index(drop=True), test.reset_index(drop=True)

# All model imputation options ->  "RandomForest","ExtraTrees","GBR","KNN","XGBoost","Lightgbm","Catboost"

hparameters=mli.imputer_parameters()
## Customizing parameters settings

hparameters["GBR"]["n_estimators"]=15
hparameters["KNN"]["n_neighbors"]=3
print(hparameters)

# Imputation Example 1 : KNN

imputer_knn=mli.fit_imput(Dataset=train,imput_model="KNN",imputer_configs=hparameters)
train_knn=mli.transform_imput(Dataset=train,fit_configs=imputer_knn)
test_knn=mli.transform_imput(Dataset=test,fit_configs=imputer_knn)

# Imputation Example 2 : GradientBoostingRegressor

imputer_gbr=mli.fit_imput(Dataset=train,imput_model="GBR",imputer_configs=hparameters)
train_gbr=mli.transform_imput(Dataset=train,fit_configs=imputer_gbr)
test_gbr=mli.transform_imput(Dataset=test,fit_configs=imputer_gbr)
    
## Performance Evaluation Example - Imputation CrossValidation

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import atlantic as atl

## Preprocessing Data (Label Encoder)

le_fit=atl.fit_Label_Encoding(train_knn,target)
df=atl.transform_Label_Encoding(train_knn,le_fit)
df=df.reset_index(drop=True)
df[target]=df[target].astype('category')

# Replace the problematic characters with underscores in the column names
df.rename(columns=lambda x: x.replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_"), inplace=True)

leaderboard_knn_imp=mli.cross_validation(Dataset=df,
                                         target=target, 
                                         test_size=0.2,
                                         n_splits=3,
                                         models=[XGBClassifier(), RandomForestClassifier(), DecisionTreeClassifier()])


#import pickle 
#output = open("imputer_knn.pkl", 'wb')
#pickle.dump(imputer_knn, output)

