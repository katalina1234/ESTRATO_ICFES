import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn. metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import watchdog
import pickle

# importar los datos

df = pd.read_excel('BASE_FINAL.xlsx')

# seleccionar X y Y

Y = df.filter(['PUNT_GLOBAL'])
X = df.drop(['Unnamed: 0','PUNT_GLOBAL','ESTU_DEPTO_RESIDE','FAMI_EDUCACIONPADRE', 'ESTU_DEDICACIONLECTURADIARIA', 'ESTU_HORASSEMANATRABAJA','FAMI_COMECARNEPESCADOHUEVO'], axis = 1); X 

# dummys
X = pd.get_dummies(X, columns=['ESTU_GENERO'], drop_first= True)
X = pd.get_dummies(X, columns=['FAMI_EDUCACIONMADRE'], drop_first= True)
X = pd.get_dummies(X, columns=['FAMI_TIENEINTERNET'], drop_first= True)
X = pd.get_dummies(X, columns=['FAMI_TIENECOMPUTADOR'], drop_first= True)
X = pd.get_dummies(X, columns=['COLE_BILINGUE'], drop_first= True)
X = pd.get_dummies(X, columns=['COLE_NATURALEZA'], drop_first= True)

#  train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .25, random_state = 20102021)


# optimizar hiperparametros
param = {'n_estimators': [2,4],
         'max_features': [4,6],
         'max_depth': [2,3]}
grid = GridSearchCV(RandomForestRegressor(), param, cv = 5)
grid.fit(X_train,Y_train)

# ajuste del modelo
reg_RF = RandomForestRegressor(**grid.best_params_)
reg_RF.fit(X_train, Y_train)

y_train_fit = reg_RF.predict(X_train)
y_pred = reg_RF.predict(X_test)

print(y_train_fit)
print(y_pred)
print(mean_absolute_error(Y_train, y_train_fit))
print(mean_absolute_error(Y_test, y_pred))

# guardar el modelo

rf_pickle = open('rf_reg.pickle','wb')
pickle.dump(reg_RF, rf_pickle) 

rf_pickle.close()