
#%%
#os library version
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

SEED = 5
#%%
# read full data
fdat = pd.read_csv("data/full_filtered_dat.csv")
fdat.season = fdat.season.map(str)
# separate train and test data
X = fdat.drop("imdb_rating", axis=1)
y = fdat.imdb_rating


# write raw files 
Path(f"./data/seed_{SEED}").mkdir(parents=True, exist_ok=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=SEED)
X_train.to_csv(f"data/SEED_{SEED}/X_train_raw.csv")
X_test.to_csv(f"data/SEED_{SEED}/X_test_raw.csv")
y_train.to_csv(f"data/SEED_{SEED}/y_train_raw.csv")
y_test.to_csv(f"data/SEED_{SEED}/y_test_raw.csv")

# create pipeline for scalers
std_scale = Pipeline([('standard', StandardScaler())])
minmax_scale = Pipeline([('minmax', MinMaxScaler())])

# select columns that require scaling
scale_col = X_train.iloc[:,0:24].select_dtypes(include=np.number).columns.tolist()

minmax_prep = ColumnTransformer(
        remainder='passthrough',
        transformers=[
            ('minmax', minmax_scale , scale_col),
        ])

std_prep = ColumnTransformer(
        remainder='passthrough',
        transformers=[
            ('std', std_scale , scale_col),
        ])

#%%
# fit scaler on training data
minmax_prep.fit(X_train)
min_test = pd.DataFrame(minmax_prep.transform(X_test))
max_train = pd.DataFrame(minmax_prep.transform(X_train))
t.columns = X_train.columns
#%%
