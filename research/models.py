from utils import*
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import numpy as np

def random_forest(X_train,Y_train,X_test):
    rf=RandomForestRegressor(n_estimators=10,random_state=42)
    rf.fit(X_train,Y_train)
    preds=rf.predict(X_test)
    return preds

def gradient_boosting(X_train,Y_train,X_test):
    #using GB for training Y2
    # Training Y1
    #gb1=HistGradientBoostingRegressor(learning_rate=1,max_depth=6,random_state=42,max_iter=300,max_leaf_nodes=40)
    #gb1.fit(X_train,Y_train['Y1'])  
    #preds1=gb1.predict(X_test)

    # Training Y2
    gb=HistGradientBoostingRegressor(learning_rate=1,max_depth=6,random_state=42,max_iter=300,max_leaf_nodes=40)
    gb.fit(X_train,Y_train['Y2'])
    preds=gb.predict(X_test)
    return preds

def xg_boosting(X_train,Y_train,X_test) :
    #Training Y2
    xg=xgb.XGBRegressor(n_estimators=750, learning_rate=0.05, max_depth=4, random_state=42)
    xg.fit(X_train,Y_train['Y2'])
    preds=xg.predict(X_test)
    return preds

def regression(X_train,Y_train,X_test):
    #Training Y1 with G
    lr=LinearRegression()
    lr.fit(X_train[['G']],Y_train['Y1'])
    preds=lr.predict(X_test[['G']])
    return preds

def mlp_regressor(X_train,Y_train,X_test) :
    # Training Y1 with G
    mlp=MLPRegressor(random_state=1,max_iter=2000,tol=0.01,hidden_layer_sizes=(10,5),activation='relu',solver='adam',validation_fraction=0.1)
    mlp.fit(X_train,Y_train['Y1'])
    preds=mlp.predict(X_test)
    return preds