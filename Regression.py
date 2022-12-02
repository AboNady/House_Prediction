import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE

dataset = pd.read_csv(r'C:\Users\NADY\Desktop\intern\house-prices-advanced-regression-techniques\DD.csv'  )
datasetY = pd.read_csv(r'C:\Users\NADY\Desktop\intern\house-prices-advanced-regression-techniques\Y.csv'  )




norm_data = preprocessing.normalize(datasetY, axis=0)
datasetY  = pd.DataFrame(norm_data,columns=[datasetY.columns])
print(datasetY)



lineaReg = LinearRegression().fit(dataset, datasetY)

print('Linear Reg')
print ( lineaReg.score(dataset, datasetY) )
rmse = np.sqrt(MSE(datasetY, lineaReg.predict(dataset) ))
print("RMSE : % f" %(rmse))
print('---------------------------------------')

  
randomF = RandomForestRegressor(n_estimators = 19, random_state = 0)
randomF.fit(dataset, datasetY)

print('Random Forest')
print(randomF.score(dataset, datasetY))
rmseF = np.sqrt(MSE(datasetY, randomF.predict(dataset) ))
print("RMSE : % f" %(rmseF))
print('---------------------------------------')

xgbR = xg.XGBRegressor(objective ='reg:linear', n_estimators = 20, seed = 123)
xgbR.fit(dataset, datasetY)
pred = xgbR.predict(dataset)

print('XGB Reg')
print(xgbR.score(dataset, datasetY))
rmsexg = np.sqrt(MSE(datasetY, pred))
print("RMSE : % f" %(rmsexg))








