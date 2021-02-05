# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:38:53 2019

@author: 
"""



##############################################################################
#
#    ALGORITHMIE DU BIG DATA
#
##############################################################################


#
# QUESTION 0 - IMPORTATION DES PACKAGES ET LIBRAIRIES UTILISEES PAR LA SUITE
# 


#import des librairies

#sys et numpy pour les calculs matriciels
import sys
import numpy as np
#pandas pour lecture donnees
import pandas as pd

#
# QUESTION 1 - IMPORT DU JEU DE DONNEES
# 


### Q1.1 - Indiquer le dossier et le fichier cible

dossier="C:/Users/annea/Desktop/TRAVAIL T2/UE10 BIG DATA/Section 2 - Algorithmes du Big Data/Section 2 - TP/TP Big Data Python/train_echantillon.csv"


### Q1.2 - Importer les jeux de données complets et échantillonnés
###        Prediction du prix du taxi à New York - https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data


# ---------- Utiliser une librairie usuelle (version de fichier échantillonnée)

UsualData= pd.read_csv('C:/Users/annea/Desktop/TRAVAIL T2/UE10 BIG DATA/Section 2 - Algorithmes du Big Data/Section 2 - TP/TP Big Data Python/train_echantillon.csv')

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory) (version complète du fichier)

#CODE


#
# QUESTION 2 - PREPARATION/NETTOYAGE DU JEU DE DONNEES
# 


### Q2.1 - Nettoyer et préparer les données


# Enlever les valeurs incorrectes ou manquantes (si pertinent)

# ---------- Utiliser une librairie usuelle

from sklearn import preprocessing

#on enleve les variables manquantes avec fonction dropna
UsualData_clean=UsualData.dropna()
#Il reste 5542347 observations, 
#au lieu des 5542385 observations de départ
#38 observations manquantes ont été supprimées

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE



# Ne garder que les variables de géolocalisation (pour le jeu de données en entrée) et
# la variable "fare_amount" pour la sortie


# ---------- Utiliser une librairie usuelle

variables=["fare_amount","pickup_longitude","pickup_latitude",
                               "dropoff_longitude", "dropoff_latitude"]
UsualData_clean=pd.DataFrame(UsualData_clean,columns=variables,dtype='float')

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Obtenir les caractéristiques statistiques de base des variables d'entrée et de sortie
# (par exemple, min, moyenne, mdéiane, max) et filter les valeurs aberrantes


# ---------- Utiliser une librairie usuelle


import pandas as pd
caracteristiques_variables=UsualData_clean.describe()
print(caracteristiques_variables)

"""
        fare_amount  pickup_longitude  ...  dropoff_longitude  dropoff_latitude
count  5.542347e+06      5.542347e+06  ...       5.542347e+06      5.542347e+06
mean   1.133738e+01     -7.250974e+01  ...      -7.250740e+01      3.991951e+01
std    9.783154e+00      1.275195e+01  ...       1.240406e+01      9.764199e+00
min   -1.000000e+02     -3.383285e+03  ...      -3.366536e+03     -3.481141e+03
25%    6.000000e+00     -7.399206e+01  ...      -7.399140e+01      4.073404e+01
50%    8.500000e+00     -7.398180e+01  ...      -7.398016e+01      4.075316e+01
75%    1.250000e+01     -7.396708e+01  ...      -7.396364e+01      4.076812e+01
max    6.981600e+02      3.239513e+03  ...       3.429641e+03      3.375202e+03

[8 rows x 5 columns]
"""
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Visualiser les distributions des variables d'entrée et de sortie (histogramme, pairplot)


# ---------- Utiliser une librairie usuelle

#CODE
import matplotlib.pyplot as plt
#plusieurs histogrammes avec boucle for sur les noms de variables
for var in  variables:
    UsualData_clean[var].plot.hist()
    plt.title(var)
    plt.show()




# Séparer la variable à prédire ("fare_amount") des autres variables d'entrée
# Créer un objet avec variables d'entrée et un objet avec valeurs de sortie (i.e. "fare_amount")



# ---------- Utiliser une librairie usuelle

#copier les donnees
input_var=variables.copy()
#enlever la variable cible
input_var.remove("fare_amount")

#creer les variables X et y
X,y=UsualData_clean[input_var],UsualData_clean["fare_amount"]




# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)


#CODE


# Standardiser la matrice d'entrée et les vecteurs de sortie (créer un nouvel objet)


# ---------- Utiliser une librairie usuelle

#on standardise les valeurs avec la fonction scale de la sous librairie 
#preprocessing de sklearn

X_scaled=preprocessing.scale(X)
y_scaled=preprocessing.scale(y)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)


#CODE







#
# QUESTION 3 - CLUSTERING DU JEU DE DONNEES
# 




### Q3.1 - Réaliser un clustering k-means sur les données d'entrée standardisées


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE



### Q3.2 - Tracer la figure de l'inertie intraclusters et du R² en fonction du nombre de  clusters


# ---------- Utiliser une librairie usuelle

#CODE





### Q3.3 - A partir de combien de clusters on peut dire que partitionner n'apporte plus 
###        grand chose? Pourquoi?



#REPONSE ECRITE (3 lignes maximum)





### Q3.4 - Comment pouvez-vous qualifier les clusters obtenus selon les variables originales?
###        Par exemple, y a-t-il des clusters selon la localisation ? 




#REPONSE ECRITE (3 lignes maximum)



### Q3.5 - Visualiser les clusters avec des couleurs différentes sur un 'pairplot' avec plusieurs variables


# ---------- Utiliser une librairie usuelle


#CODE









#
# QUESTION 4 - ANALYSE EN COMPOSANTES PRINCIPALES (ACP) POUR SIMPLIFIER LE JEU DE DONNEES
# 



### Q4.1 - Faire une ACP sur le jeu de données standardisé


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


### Q4.2 - Réaliser le diagnostic de variance avec un graphique à barre (barchart)

 

# ---------- Utiliser une librairie usuelle


#CODE




### Q4.3 - Combien de composantes doit-on garder? Pourquoi?
       


#REPONSE ECRITE (3 lignes maximum)




### Q4.4 - Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premières CP
###        Sélectionner éventuellement un sous-échantillon de points pour faciliter la visualisation

 

# ---------- Utiliser une librairie usuelle


#CODE




### Q4.5 - Comment les variables initiales se situent-elles par rapport aux 2 premières CP? 


#REPONSE ECRITE (3 lignes maximum)











#
# QUESTION 5 - REGRESSION LINEAIRE
# 



### Q5.1 - Mener une régression linéaire de la sortie "fare_amount" 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# ---------- Utiliser une librairie usuelle

from sklearn import linear_model

regr= linear_model.LinearRegression()

regr.fit(X_scaled,y_scaled)

# Evaluation du modele
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
y_predict = regr.predict(X_scaled)

#Calcul de l erreur quadratique moyenne (rmse)
rmse = (np.sqrt(mean_squared_error(y_scaled,y_predict)))
#rmse=0.9999373484290106

#calcul du R²
r2=r2_score(y_scaled,y_predict)
print(r2)
#R²=0.0001252992167616318

print(regr.intercept_)
#intercept=9.760390606573518e-16
print(regr.coef_)
#coefficients des variables explicatives "pickup_longitude","pickup_latitude",
#dropoff_longitude", "dropoff_latitude"
#[ 0.00351443 -0.00107122  0.00582572 -0.00306093]

#Pour obtenir les pvalues, on utilise statsmodels
#Remarque : on invese X et y dans la spécification du modèle pour cette librairie.
import statsmodels.api as sm
model = sm.OLS(y_scaled,X_scaled)
results = model.fit()
# Avec  statsmodel, on a une sortie qui ressemble beaucoup à celle de R
print(results.summary())

"""
 OLS Regression Results                                
=======================================================================================
Dep. Variable:                      y   R-squared (uncentered):                   0.000
Model:                            OLS   Adj. R-squared (uncentered):              0.000
Method:                 Least Squares   F-statistic:                              173.6
Date:                Fri, 05 Feb 2021   Prob (F-statistic):                   5.42e-149
Time:                        18:54:59   Log-Likelihood:                     -7.8639e+06
No. Observations:             5542347   AIC:                                  1.573e+07
Df Residuals:                 5542343   BIC:                                  1.573e+07
Df Model:                           4                                                  
Covariance Type:            nonrobust                                                  
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.0035      0.001      5.352      0.000       0.002       0.005
x2            -0.0011      0.001     -1.979      0.048      -0.002   -1.01e-05
x3             0.0058      0.001      8.670      0.000       0.005       0.007
x4            -0.0031      0.001     -6.022      0.000      -0.004      -0.002
==============================================================================
Omnibus:                  5086820.045   Durbin-Watson:                   2.002
Prob(Omnibus):                  0.000   Jarque-Bera (JB):        508494903.927
Skew:                           4.087   Prob(JB):                         0.00
Kurtosis:                      49.207   Cond. No.                         3.23
==============================================================================
"""



# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


### Q5.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?

#var1=pickup_longitude,var2=pickup_latitude,var3=dropoff_longitude,var4=dropoff_latitude
#Les trois variables explicatives les plus significatives sont, par ordre d'importance
#decroissante: var3=dropoff_longitude, var1=pickup_longitude, var4=dropoff_latitude
#Elles ont toutes les trois une pvalue nulle donc <5%
#La variable var2=pickup_latitude est mois significative avec une pvalue de 0.048
#Elle est neanmoins significative a un seuil de 5% 



### Q5.3 - Prédire le prix de la course en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Réaliser la régression linéaire sur l'échantillon d'apprentissage, tester plusieurs valeurs
# de régularisation (hyperparamètre de la régression linéaire) et la qualité de prédiction sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE

# Calculer le RMSE et le R² sur le jeu de test.



# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE

# Quelle est la qualité de la prédiction sur le jeu de test ?


#REPONSE ECRITE (3 lignes maximum)








#
# QUESTION 6 - REGRESSION LOGISTIQUE
# 



### Q6.1 - Mener une régression logisitique de la sortie "fare_amount" (après binarisation selon la médiane) 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# Créer la sortie binaire 'fare_binaire' en prenant la valeur médiane de "fare_amount" comme seuil


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Mener la régression logistique de "fare_binaire" en fonction des entrées standardisées


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE




### Q6.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?



#REPONSE ECRITE (3 lignes maximum)



### Q6.3 - Prédire la probabilité que la course soit plus élevée que la médiane
#           en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Réaliser la régression logistique sur l'échantillon d'apprentissage et en testant plusieurs valeurs
# de régularisation (hyperparamètre de la régression logistique) sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Calculer la précision (accuracy) et l'AUC de la prédiction sur le jeu de test.



# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Quelle est la qualité de la prédiction sur le jeu de test ?


#REPONSE ECRITE (3 lignes maximum)







#
# QUESTION 7 - RESEAU DE NEURONES (QUESTION BONUS)
# 



### Q7.1 - Mener une régression de la sortie "fare_amount" en fonction de l'entrée (mise à l'échelle), 
###       sur tout le jeu de données, avec un réseau à 2 couches cachées de 10 neurones chacune



# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE



### Q7.2 - Prédire le prix de la course en fonction de nouvelles entrées avec le réseau de neurones entraîné


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Réaliser la régression avec réseau de neurones sur l'échantillon d'apprentissage et en testant plusieurs 
# nombre de couches et de neurones par couche sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Calculer le RMSE et le R² de la meilleure prédiction sur le jeu de test.


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE

# Quelle est la qualité de la prédiction sur le jeu de test ? Comment se compare-t-elle à la régression linéaire?


#REPONSE ECRITE (3 lignes maximum)

