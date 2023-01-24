# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import LabelEncoder

trainData = pd.read_csv('train_dosyasi.csv')
testData = pd.read_csv('test.csv')

def ml(trainData, testData, model=[]):
    le=LabelEncoder()
    trainData['urun_kategorisi']=le.fit_transform(trainData['urun_kategorisi'])
    trainData['urun']=le.fit_transform(trainData['urun'])
    trainData['sehir']=le.fit_transform(trainData['sehir'])
    trainData['market']=le.fit_transform(trainData['market'])
    trainData['urun_uretim_yeri']=le.fit_transform(trainData['urun_uretim_yeri'])
    trainData['tarih']=le.fit_transform(trainData['tarih'])

    testData['urun_kategorisi']=le.fit_transform(testData['urun_kategorisi'])
    testData['urun']=le.fit_transform(testData['urun'])
    testData['sehir']=le.fit_transform(testData['sehir'])
    testData['market']=le.fit_transform(testData['market'])
    testData['urun_uretim_yeri']=le.fit_transform(testData['urun_uretim_yeri'])
    testData['tarih']=le.fit_transform(testData['tarih'])

    X = trainData.drop(["urun_fiyati"], axis=1)
    y = trainData["urun_fiyati"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
    
    for i in range(len(model)):
        if(model[i] == MLPRegressor):
            models = model[i](max_iter=5000)
            variable = 'mlp.csv'
        elif(model[i] == DecisionTreeRegressor):
            models = model[i](max_depth=5)
            variable = 'decisionTree.csv'
        elif(model[i] == KNeighborsRegressor):
            models = model[i](n_neighbors=10, n_jobs=-1)
            variable = 'knn.csv'
        elif(model[i] == RandomForestRegressor):
            models = model[i](random_state=0, n_estimators=5000)
            variable = 'randomForest.csv'
        elif(model[i] == GradientBoostingRegressor):
            models = model[i](random_state=0, n_estimators = 5000, learning_rate=0.1)
            variable = 'gradientBoosting.csv'
        else:
            models = model[i](random_state=0,n_estimators = 5000)
            variable = 'xgb.csv'
            
        models.fit(X_train, y_train)
        y_pred = models.predict(X_test)
        rmse = (np.sqrt(MSE(y_test, y_pred)))
        print("RMSE:{} ".format(models),rmse)
        
        pred = models.predict(testData)
        real=pd.DataFrame(pred,columns=["urun_fiyati"])

        real['id']=np.arange(0,real.shape[0])

        cols = real.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        real = real[cols]

        real.to_csv(variable,index=False)

model = [DecisionTreeRegressor,
          KNeighborsRegressor,
          MLPRegressor,
          RandomForestRegressor,
          GradientBoostingRegressor,
          XGBRegressor]

ml(trainData, testData, model=model)
