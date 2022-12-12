from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE


dataset = pd.read_csv(r'C:\Users\NADY\Desktop\train.csv'  )
datasetY = dataset['SalePrice']
dataset = dataset.drop(['PoolQC','Fence','SalePrice','MiscFeature','FireplaceQu','Alley','Id' ], axis = 1) 

datasetY = np.array( datasetY )
datasetY = datasetY.reshape(-1, 1)

y = datasetY #returns a numpy array
datasetY = pd.DataFrame( preprocessing.MinMaxScaler().fit_transform(y))

lb = LabelEncoder()
dataset['LotShape']      = lb.fit_transform(dataset['LotShape'])
dataset['MSZoning'] = lb.fit_transform(dataset['MSZoning'])

dataset['Street']      = lb.fit_transform(dataset['Street'])

dataset['LandContour']      = lb.fit_transform(dataset['LandContour'])
dataset['LotConfig'] = lb.fit_transform(dataset['LotConfig'])

dataset['LandSlope']      = lb.fit_transform(dataset['LandSlope'])
dataset['Condition1'] = lb.fit_transform(dataset['Condition1'])

dataset['Condition2']      = lb.fit_transform(dataset['Condition2'])
dataset['BldgType'] = lb.fit_transform(dataset['BldgType'])

dataset['HouseStyle']      = lb.fit_transform(dataset['HouseStyle'])
dataset['RoofStyle'] = lb.fit_transform(dataset['RoofStyle'])

dataset['RoofMatl']      = lb.fit_transform(dataset['RoofMatl'])
dataset['Exterior1st'] = lb.fit_transform(dataset['Exterior1st'])
#--------------


dataset['Exterior2nd']      = lb.fit_transform(dataset['Exterior2nd'])
dataset['MasVnrType'] = lb.fit_transform(dataset['MasVnrType'])

dataset['ExterQual']      = lb.fit_transform(dataset['ExterQual'])
dataset['ExterCond'] = lb.fit_transform(dataset['ExterCond'])

dataset['Foundation']      = lb.fit_transform(dataset['Foundation'])
dataset['BsmtQual'] = lb.fit_transform(dataset['BsmtQual'])

dataset['BsmtCond']      = lb.fit_transform(dataset['BsmtCond'])
dataset['BsmtExposure'] = lb.fit_transform(dataset['BsmtExposure'])

dataset['BsmtFinType1']      = lb.fit_transform(dataset['BsmtFinType1'])
dataset['BldgType'] = lb.fit_transform(dataset['BldgType'])

dataset['BsmtFinType2']      = lb.fit_transform(dataset['BsmtFinType2'])
dataset['Heating'] = lb.fit_transform(dataset['Heating'])


dataset['HeatingQC'] = lb.fit_transform(dataset['HeatingQC'])

dataset['CentralAir']      = lb.fit_transform(dataset['CentralAir'])
dataset['Electrical'] = lb.fit_transform(dataset['Electrical'])

#--------------


dataset['KitchenQual']      = lb.fit_transform(dataset['KitchenQual'])
dataset['Functional'] = lb.fit_transform(dataset['Functional'])

dataset['GarageType'] = lb.fit_transform(dataset['GarageType'])

dataset['GarageFinish']      = lb.fit_transform(dataset['GarageFinish'])
dataset['GarageQual'] = lb.fit_transform(dataset['GarageQual'])

dataset['GarageCond']      = lb.fit_transform(dataset['GarageCond'])
dataset['PavedDrive'] = lb.fit_transform(dataset['PavedDrive'])

dataset['SaleType']      = lb.fit_transform(dataset['SaleType'])
dataset['SaleCondition'] = lb.fit_transform(dataset['SaleCondition'])


dataset['Utilities']      = lb.fit_transform(dataset['Utilities'])
dataset['Neighborhood'] = lb.fit_transform(dataset['Neighborhood'])



dataset = dataset.fillna(dataset.mean())




x = dataset.values #returns a numpy array
dataset = pd.DataFrame( preprocessing.MinMaxScaler().fit_transform(x))



Xtrain, Xtest, Ytrain, Ytest = train_test_split( dataset, datasetY, test_size=0.3, random_state=6)



lineaReg = LinearRegression().fit(Xtrain, Ytrain)

print('Linear Reg')
print ( lineaReg.score(Xtest, Ytest) )
rmse = np.sqrt(MSE(Ytest, lineaReg.predict(Xtest) ))
print("RMSE : % f" %(rmse))
print('---------------------------------------')

  
randomF = RandomForestRegressor(n_estimators = 23, random_state = 0, criterion= 'squared_error', max_features = 97)
randomF.fit(Xtrain, Ytrain.values.ravel() )

print('Random Forest')
print(randomF.score(Xtest, Ytest))
rmse = np.sqrt(MSE(Ytest, randomF.predict(Xtest) ))

print("RMSE : % f" %(rmse))
print('---------------------------------------')

xgbR = xg.XGBRegressor(objective ='reg:linear',verbosity = 0, n_estimators = 20, seed = 1, booster = 'dart', eta = 0.2, max_depth = 6,
                       reg_lambda = 0.005)
xgbR.fit(Xtrain, Ytrain)
pred = xgbR.predict(Xtest)

print('XGB Reg')
print(xgbR.score(Xtest, Ytest))
rmsexg = np.sqrt(MSE(Ytest, pred))
print("RMSE : % f" %(rmsexg))

print('---------------------------------------')



