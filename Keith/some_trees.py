#%%
import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#%%

X_train = pd.read_csv("../deliverables/data/seed_5/X_train_std.csv")
X_test  = pd.read_csv("../deliverables/data/seed_5/X_test_std.csv")
y_train = np.array(pd.read_csv("../deliverables/data/seed_5/y_train.csv")).flatten()
y_test  = np.array(pd.read_csv("../deliverables/data/seed_5/y_test.csv")).flatten()
# %%

dt_1 = DecisionTreeRegressor()
leave_1 = LeaveOneOut()

scores = cross_val_score(dt_1, X_train, y_train, scoring='neg_mean_squared_error',
                         cv= leave_1, n_jobs=-1)


np.sqrt(np.mean(np.absolute(scores)))


#%%
dt_1.fit(X_train, y_train)
sns.scatterplot(y_test, dt_1.predict(X_test))
# %%
dt_1 = DecisionTreeRegressor(random_state = 5)
leave_1 = LeaveOneOut()

scores = cross_val_score(dt_1, X_train, y_train, scoring='neg_mean_squared_error',
                         cv= leave_1, n_jobs=-1)


np.sqrt(np.mean(np.absolute(scores)))

#%%
rf_1 = RandomForestRegressor(random_state = 5)

scores = cross_val_score(rf_1, X_train, y_train.flatten(), scoring="neg_mean_squared_error",
                         cv= 3, n_jobs=-1)

rf_scores = np.sqrt(np.mean(np.absolute(scores)))
rf_scores

#%%
rf_1.fit(X_train, y_train)
sns.scatterplot(y_test, rf_1.predict(X_test))

# %%
data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)

params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 7, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=150,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=5)

cv_results
# %%
xgb_2 = xgb.XGBRegressor(objective = "reg:squarederror", 
                         learning_rate = 0.1,
                         random_state = 11)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.5, 0.75, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8],
        'max_depth': [2, 3, 4]
        }

random_search = RandomizedSearchCV(xgb_2, param_distributions=params, 
                            scoring="neg_mean_squared_error", n_jobs=3, 
                            cv=7, verbose=3)

random_search.fit(X_train, y_train)
random_search.best_params_
# %%
params = {
    "objective":"reg:squarederror",
    'learning_rate': 0.1, 
    'subsample': .5,
    'min_child_weight': 10,
    'max_depth': 3,
    'gamma': 0.5,
    'colsample_bytree': 0.8}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=150,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=5)

cv_results
# %%
fin_xgb = xgb.XGBRegressor(    
    objective = "reg:squarederror",
    learning_rate = 0.1, 
    subsample = .5,
    min_child_weight = 10,
    max_depth = 3,
    gamma = 0.5,
    colsample_bytree = 0.8)
fin_xgb.fit(X_train, y_train)
fin_xgb.predict(X_test)
r2_score(fin_xgb.predict(X_test), y_test)
# %%
mean_squared_error(y_test, fin_xgb.predict(X_test))

#%%
sns.scatterplot(y_test.tolist(), fin_xgb.predict(X_test))

# %%

