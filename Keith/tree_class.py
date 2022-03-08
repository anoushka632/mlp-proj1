import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix

#%%

X_train = pd.read_csv("../deliverables/data/seed_55/X_train_std.csv")
X_test  = pd.read_csv("../deliverables/data/seed_55/X_test_std.csv")
y_train = np.array(pd.read_csv("../deliverables/data/seed_55/y_train_cl.csv")).flatten()
y_test  = np.array(pd.read_csv("../deliverables/data/seed_55/y_test_cl.csv")).flatten()

#%%
rf_c = RandomForestClassifier(random_state = 76)

scores = cross_val_score(rf_c, X_train, y_train.flatten(), scoring='accuracy',
                         cv= 4, n_jobs=-1)

rf_scores = np.sqrt(np.mean(np.absolute(scores)))
rf_scores
# %%
xgb_2 = xgb.XGBClassifier(random_state = 57)


params = {
        "eta": [0.05, .1, .15, .2],
        'min_child_weight': [1, 5, 10],
        'subsample': [0.5, 0.75, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8],
        'max_depth': [2, 4,6]
        }

random_search = RandomizedSearchCV(xgb_2, param_distributions=params, 
                            scoring="neg_mean_squared_error", n_jobs=3, 
                            cv=7, verbose=3)

grid_search = GridSearchCV(xgb_2, param_grid=params, 
                            scoring="r2", n_jobs=3, 
                            cv=4, verbose=3)

random_search.fit(X_train, y_train)
random_search.best_params_

#%%
params_ = {
    "objective":"multi:softprob",
    'learning_rate': 0.1, 
    'subsample': .75,
    'min_child_weight': 5,
    'max_depth': 6,
    'gamma': 0.3,
    "alpha": 1,
    'colsample_bytree': 0.4}

fin_xgb = xgb.XGBClassifier(    
    objective = "multi:softprob",
    learning_rate = 0.1, 
    subsample = .75,
    min_child_weight = 5,
    max_depth = 6,
    gamma = .3,
    alpha = 1,
    colsample_bytree = 0.4,
    eval_mertic = "merror")
fin_xgb.fit(X_train, y_train)
fin_xgb.predict(X_test)
#r2_score(fin_xgb.predict(X_test), y_test)
# %%

confusion_matrix(y_true = y_test,y_pred = fin_xgb.predict(X_test))
# %%
