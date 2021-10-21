import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import pickle

#importar los datos
df = pd.read_excel('BASE_FINAL.xlsx')
Nombres = df.columns
Genero_dummy = pd.get_dummies(df[Nombres[2]], prefix = Nombres[1])
Est_padre = pd.get_dummies(df[Nombres[4]], prefix = Nombres[4])
Est_madre = pd.get_dummies(df[Nombres[5]], prefix = Nombres[5])
Internet =  pd.get_dummies(df[Nombres[6]], prefix = Nombres[6])
Compurador = pd.get_dummies(df[Nombres[7]], prefix = Nombres[7])
Lectura = pd.get_dummies(df[Nombres[8]], prefix = Nombres[8])
Trabaja = pd.get_dummies(df[Nombres[9]], prefix = Nombres[9])
Colegio_B =  pd.get_dummies(df[Nombres[10]], prefix = Nombres[10])
Carne = pd.get_dummies(df[Nombres[11]], prefix = Nombres[11])
Oficial = pd.get_dummies(df[Nombres[12]], prefix = Nombres[12])

frames = [df['EDAD'], Genero_dummy,  Est_padre, Est_madre, Internet, Compurador, Lectura, Trabaja, Colegio_B, Carne, Oficial]
X_r = pd.concat(frames, axis =1)
Y_r = df.filter(['PUNT_GLOBAL'])
del X_r['EDAD_F'] , X_r['FAMI_EDUCACIONPADRE_Ninguno'] , X_r['FAMI_EDUCACIONMADRE_Ninguno'], X_r['FAMI_TIENEINTERNET_No'], X_r[ 'FAMI_TIENECOMPUTADOR_No'], X_r['ESTU_DEDICACIONLECTURADIARIA_No leo por entretenimiento'], X_r['ESTU_HORASSEMANATRABAJA_0'], X_r['COLE_BILINGUE_S'], X_r['FAMI_COMECARNEPESCADOHUEVO_Todos o casi todos los dias'] , X_r['COLE_NATURALEZA_NO OFICIAL']
X_train, X_test, Y_train, Y_test = train_test_split(X_r, Y_r, test_size = .25, random_state = 20102021)
X_train.shape, X_test.shape, Y_train.shape , Y_test.shape
modelo_reg = LinearRegression(fit_intercept= True) 
modelo_reg.fit(X_train, Y_train)
Y_pred_train = modelo_reg.predict(X_train)
Y_pred_test = modelo_reg.predict(X_test)
def metricas(y_real, y_estimado):
  print(f"Error Cuadratico Medio: {mean_squared_error(y_real, y_estimado)}")
  print(f"Error Absoluto Medio: {mean_absolute_error(y_real, y_estimado)}")
  print(f"Raíz del error cuadrático medio: {np.sqrt(mean_squared_error(y_real, y_estimado))}")
  print("Métricas del entrenamiento", end = "\n")
print("---"*10)
metricas(Y_train,Y_pred_train)
print("---"*10)
print("Métricas del Testeo", end = "\n")
print("---"*10)
metricas(Y_test,Y_pred_test)
param = {'alpha':[1,10,20,30,50,100,200]}
model = Ridge()
grid = GridSearchCV(model, param, cv = 5)
grid.fit(X_train, Y_train)
regresion_final = Ridge(**grid.best_params_)
regresion_final.fit(X_train, Y_train)
y_train_fit = regresion_final.predict(X_train)
y_pred = regresion_final.predict(X_test)
print(mean_absolute_error(Y_train, y_train_fit))
print(mean_absolute_error(Y_test, y_pred))
Y = df.filter(['PUNT_GLOBAL'])
X = df.drop(['Unnamed: 0','PUNT_GLOBAL','ESTU_DEPTO_RESIDE','FAMI_EDUCACIONPADRE', 'ESTU_DEDICACIONLECTURADIARIA', 'ESTU_HORASSEMANATRABAJA','FAMI_COMECARNEPESCADOHUEVO'], axis = 1); X
X = pd.get_dummies(X, columns=['ESTU_GENERO'], drop_first= True)
X = pd.get_dummies(X, columns=['FAMI_EDUCACIONMADRE'], drop_first= True)
X = pd.get_dummies(X, columns=['FAMI_TIENEINTERNET'], drop_first= True)
X = pd.get_dummies(X, columns=['FAMI_TIENECOMPUTADOR'], drop_first= True)
X = pd.get_dummies(X, columns=['COLE_BILINGUE'], drop_first= True)
X = pd.get_dummies(X, columns=['COLE_NATURALEZA'], drop_first= True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .25, random_state = 20102021)
param = {'n_estimators': [2,4],
         'max_features': [4,6],
         'max_depth': [2,3]}
grid = GridSearchCV(RandomForestRegressor(), param, cv = 5)
mod = RandomForestRegressor()
grid.fit(X_train,Y_train)
reg_RF = RandomForestRegressor(**grid.best_params_)
reg_RF.fit(X_train, Y_train)
y_train_fit = reg_RF.predict(X_train)
y_pred = reg_RF.predict(X_test)
metricas(Y_train,y_train_fit)
metricas(Y_test,y_pred)

import xgboost as xgb

X = df.drop(['Unnamed: 0','PUNT_GLOBAL','ESTU_DEPTO_RESIDE','FAMI_EDUCACIONPADRE', 'ESTU_DEDICACIONLECTURADIARIA', 'ESTU_HORASSEMANATRABAJA','FAMI_COMECARNEPESCADOHUEVO'], axis = 1); X
Y = df.filter(['PUNT_GLOBAL'])
X = pd.get_dummies(X, columns=['ESTU_GENERO'], drop_first= True)
X = pd.get_dummies(X, columns=['FAMI_EDUCACIONMADRE'], drop_first= True)
X = pd.get_dummies(X, columns=['FAMI_TIENEINTERNET'], drop_first= True)
X = pd.get_dummies(X, columns=['FAMI_TIENECOMPUTADOR'], drop_first= True)
X = pd.get_dummies(X, columns=['COLE_BILINGUE'], drop_first= True)
X = pd.get_dummies(X, columns=['COLE_NATURALEZA'], drop_first= True)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=1515)
param1 = {'n_estimators': [4,6,8],
          'objective':['reg:squarederror'],
          'learning_rate':[0.1,0.5,0.8],
          'max_depth':[3,4,5]
          }
grid1 = GridSearchCV(xgb.XGBRegressor(), param1, cv = 5)
grid1.fit(X_train,Y_train)
reg_XGB = xgb.XGBRegressor(**grid1.best_params_)
reg_XGB.fit(X_train, Y_train)
y_train_fit = reg_XGB.predict(X_train)
y_pred = reg_XGB.predict(X_test)
print('Métricas del entrenamiento',end='\n')
print('---'*10)
metricas(Y_train, y_train_fit)
print('---'*10)
print('Métricas del testeo')
print('---'*10)
metricas(Y_test, y_pred)

# importar los datos


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

#  trait_test_split

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

print(mean_absolute_error(Y_train, y_train_fit))
print(mean_absolute_error(Y_test, y_pred))


X = df.filter(['EDAD','ESTU_GENERO','ESTU_DEPTO_RESIDE','FAMI_EDUCACIONPADRE','FAMI_EDUCACIONMADRE','FAMI_TIENEINTERNET','FAMI_TIENECOMPUTADOR',
'ESTU_DEDICACIONLECTURADIARIA', 'ESTU_HORASSEMANATRABAJA','COLE_BILINGUE', 'FAMI_COMECARNEPESCADOHUEVO','COLE_NATURALEZA'])
Y= df.filter(['PUNT_GLOBAL'])