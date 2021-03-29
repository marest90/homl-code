import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

print('version: 1.01')

#########################################################################
#Importamos el dataset

HOUSING_PATH = 'datasets/'

def load_housing_data(ruta):
	datos = pd.read_csv(ruta + 'housing.csv')
	return datos

housing = load_housing_data(HOUSING_PATH)

###########################################################################
#Analizando la data


print('Encabezado:')
print(housing.head()) #imprime el encabezado de la lista
print('##################')

print('total bedrooms:')
print(housing["total_bedrooms"]) #imprime el encabezado de la lista
print('##################')

print('Info:')
housing.info() # imprime la informacion básica. Con verbose = True tira todo y con False solo una parte
print('##################')

print('Contenido de ocean_proximity:')
print(housing["ocean_proximity"].value_counts()) #te dice que categorias existen y cuantos pertenecen a cada una
print('##################')

#Graficamos histogramas de todas las categorias
for c in housing.columns:
	print(c)
	plt.figure(c)
	plt.hist(housing[c],bins=50)
	#plt.hist(housing,bins=50, figsize=(20,15))
	plt.savefig('src-temp-docker/' + c + '.png')


print('Estadística:')
print(housing.describe())
print('##################')


##################################################################################
#Separando los sets

housing["income_cat"] = pd.cut(housing["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5]) #tomo la columna con la ganancia media y lo divido en cinco grupos, y etiqueto cada uno en un grupo
print('overall =',housing["income_cat"].value_counts() / len(housing)) #proporciones en el train set

plt.figure("income_cat")
plt.hist(housing["income_cat"])
plt.savefig('src-temp-docker/income_cat.png')

#separo de forma random
train_set, test_set = train_test_split(housing, test_size=0.2,random_state=42) #divido en train y test usando una funcion de scikit learn
print('random =',test_set["income_cat"].value_counts() / len(test_set)) #proporciones cuando elijo de forma random

#separo considerando las proporciones del income
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]
print('stratified =',strat_test_set["income_cat"].value_counts() / len(strat_test_set)) #proporciones cuando elijo con criterio


#####################################################################################
#Trabajamos con el estratificado. Visualizamos la data
housing = strat_train_set.copy()

plt.figure("mapa")
housing.plot(kind="scatter", x="longitude", y="latitude",alpha=0.1)
plt.savefig('src-temp-docker/mapa.png')

plt.figure("mapa-valor")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,s=housing["population"]/100, label="population", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)#s es para cambiar el tamaño del marcador según alguna variable elegida, c es para establecer una gama de colores según otra variable
plt.legend()
plt.savefig('src-temp-docker/mapa-valor.png')


#####################################################################################33
#Buscando correlaciones

corr_matrix = housing.corr() #mide correlacion solo lineal
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#graficamos las correlaciones entre distintos atributos
attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
plt.figure("correlaciones?")
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.savefig('src-temp-docker/correlaciones.png')


#########################################################################################
#Buscando nuevas correlaciones

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] =housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

######################################################################################
#Preparando el algoritmo

housing = strat_train_set.drop("median_house_value", axis=1)   #hace un set nuevo sin median_house_value
housing_labels = strat_train_set["median_house_value"].copy()  #solo etiquetas (median_house_value)

#Analizamos total_bedrooms que tiene espacios vacíos.
#Pueden rellenarse con: (elegimos una de las 3 opciones que hay abajo)

#housing_sintb = housing.dropna(subset=["total_bedrooms"])# option 1 elimina valores vacios
#print(housing_sintb["total_bedrooms"])
#housing_sintb = housing.drop('total_bedrooms', axis=1)# option 2 elimina toda la columna
#print(housing_sintb)
median = housing["total_bedrooms"].median() # option 3 rellena con la mediana
housing["total_bedrooms"].fillna(median, inplace=True)


##########################################################
#Encoders, transformers y pipelines

housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
#print(housing_cat_1hot)
#print(housing_cat_1hot.toarray()) #para ver como array (o sea ver el vector de cada uno)
#print(cat_encoder.categories_) #permite ver las categorias

#un pequeño transformer customizado que arma las clases discutidas anteriormente,
#rooms_per_household, population_per_household y bedrooms_per_room
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

#pipeline, llama a los anteriores transformers y los pone en serie
housing_num = housing.drop("ocean_proximity", axis=1)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])
housing_num_tr = num_pipeline.fit_transform(housing_num)

#pipeline en columnas, de texto y numéricas
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ])
housing_prepared = full_pipeline.fit_transform(housing)

#print(housing_prepared)


######################################################33
#Elección y entrenamiento del modelo

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#me quedo con las primeras 5 columnas para probar
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

#sacamos el RMSE
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print('RMSE=',lin_rmse)
#Este hizo underfitting

#Probamos con un arbol de DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
#print('RMSE=',tree_rmse)
#este dio error 0 asi que seguro esta overffitiando pero para ver que pasa vamos a usar cross validation

#Cross validation
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores) #esto divide el training set en 10 subsets

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
#display_scores(tree_rmse_scores)
#efectivamente decision tree overfittea y el resultado es peor

#random RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print('RMSE=',forest_rmse)

#Cross validation
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores) #esto divide el training set en 10 subsets
display_scores(forest_rmse_scores)


#fine tunning
#gridsearchcv hace una grilla con todos los valores de cada corrida para automatizarlas

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
grid_search.best_estimator_

#Evaluacion en el test set
final_model = lin_reg #uso uno de los modelos previamente entrenados
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test) #notar que es solo transform, no fit transform! para no modificar el test set
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,730.2
print('final error =',final_rmse)


#Ejercicio con SVM
#svm_reg = SVR(kernel='rbf',C=100,gamma=0.1, epsilon=0.1)
#svm_reg.fit(housing_prepared, housing_labels)

#me quedo con las primeras 5 columnas para probar
#print("Predictions:", svm_reg.predict(some_data_prepared))
#print("Labels:", list(some_labels))

#sacamos el RMSE
#housing_predictions = svm_reg.predict(housing_prepared)
#svm_mse = mean_squared_error(housing_labels, housing_predictions)
#svm_rmse = np.sqrt(svm_mse)
#print('RMSE=',svm_rmse)

#SVM con grid_search
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#svr = SVR()
#clf = GridSearchCV(svr, parameters)
#clf.fit(housing_prepared, housing_labels)
#GridSearchCV(estimator=SVC(),
 #            param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
#print(sorted(clf.cv_results_.keys()))

