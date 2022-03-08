import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#%%

X_train = pd.read_csv("../deliverables/data/seed_55/X_train_std.csv")
X_test  = pd.read_csv("../deliverables/data/seed_55/X_test_std.csv")
y_train = np.array(pd.read_csv("../deliverables/data/seed_55/y_train.csv")).flatten()
y_test  = np.array(pd.read_csv("../deliverables/data/seed_55/y_test.csv")).flatten()
