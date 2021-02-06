# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:38:53 2019

@author:Groupe 2
BAKOP-KAMDEM Armel
LAFAY Anne
OROZCO-HERNANDEZ Felipe

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
#Pour la régression logistique
from sklearn.preprocessing import StandardScaler
#Pour les PCA
from sklearn.decomposition import PCA

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

import dask.dataframe as dd
BigData=dd.read_csv('C:/Users/annea/Desktop/TRAVAIL T2/UE10 BIG DATA/Section 2 - Algorithmes du Big Data/Section 2 - TP/TP Big Data Python/train_echantillon.csv')

#NB: le fichier train étant trop lourd pour l'ordinateur, nous allons utiliser le fichier train_echantillon à la fois pour la partie librairie usuelle
#et pour la partie librairie big data, en donnant des noms différents aux données et en réalisant des traitements différents


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

BigData_clean=BigData.dropna()



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


X_bd,y_bd=BigData_clean[input_var],BigData_clean["fare_amount"]


# Standardiser la matrice d'entrée et les vecteurs de sortie (créer un nouvel objet)


# ---------- Utiliser une librairie usuelle

#on standardise les valeurs avec la fonction scale de la sous librairie 
#preprocessing de sklearn

X_scaled=preprocessing.scale(X)
y_scaled=preprocessing.scale(y)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)


X_scaled_bd=preprocessing.scale(X_bd)
y_scaled_bd=preprocessing.scale(y_bd)






#
# QUESTION 3 - CLUSTERING DU JEU DE DONNEES
# 




### Q3.1 - Réaliser un clustering k-means sur les données d'entrée standardisées


# ---------- Utiliser une librairie usuelle

#Implementation de k_means
#Preparation des donnees - standardisation
from sklearn.preprocessing import StandardScaler

#on utilise les donnees standardisees (_scaled)

#Modelisation usuelle (ie en local)

#K means version normale
from sklearn.cluster import KMeans
#n_clusters= nombre de clusters
#random_state= nombre aléatoire initialisant le tirage des barycentres
#fonction fit sur les donnees centrees normalisees
kmeans_model=KMeans(n_clusters=4,random_state=1).fit(X_scaled)
#sorties donc predictions de clusters en utilisant labels sur l'objet qui vient 
#d etre cree
labels_modele=kmeans_model.labels_
print(labels_modele)
#inertia donne la distance: correspond a total withinss
#variance totale dans les clusters
inertia=kmeans_model.inertia_
print(inertia)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#modelisation big data de kmeans clustering avec Dask

#fonction Kmeans de dask
from dask_ml.cluster import KMeans
kmeans_dask=KMeans(n_clusters=4)
kmeans_dask.fit_transform(X_scaled)
cluster=kmeans_dask.labels_
print(cluster)
prediction_kmeans_dask=kmeans_dask.predict(X_scaled_bd) 
#prediction de la sortie a partir de nouvelles variables d entree

#diagnostic - choix du nombre de clusters
#quand on a entraine le modele, on peut en faire le diagnostic et notamment 
#choisir le nombre de clusters optimal

for k in range (1,10):
    kmeans_model=KMeans(n_clusters=k, random_state=1).fit(X_scaled)
    labels=kmeans_model.labels_
    inertia=kmeans_model.inertia_
    print("Nombre de clusters:" + str(k)+"Inertie:"+str(inertia))
    
"""  
Nombre de clusters:1Inertie:22169387.999999993
Nombre de clusters:2Inertie:20445632.773052752
Nombre de clusters:3Inertie:18370731.792631056
Nombre de clusters:4Inertie:17466502.011337496
Nombre de clusters:5Inertie:16615631.02471132
Nombre de clusters:6Inertie:16148438.769385165
Nombre de clusters:7Inertie:15746105.325751008
Nombre de clusters:8Inertie:15518406.015413271
Nombre de clusters:9Inertie:15376890.825024331
"""
### Q3.2 - Tracer la figure de l'inertie intraclusters et du R² en fonction du nombre de  clusters


###################################
#Code de la question Q3.2 à debugger
##################################
"""
# ---------- Utiliser une librairie usuelle

#Visualisation création de clusters
data_cluster=UsualData_clean["pickup_longitude","pickup_latitude",
                               "dropoff_longitude", "dropoff_latitude"]
data_cluster["cluster"]=labels
#on prend 1000 valeurs
sample_index=np.random.randint(0,len(X_scaled),1000)

sns.pairplot(data_cluster.loc[sample_index,:],hue="cluster")
plt.show()

"""



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
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca_resultat = pca.fit_transform(X_scaled)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE
import numpy as np
import dask.array as da
from dask_ml.decomposition import PCA

dX = da.from_array(X_scaled_bd, chunks=X_scaled_bd.shape)
pca = PCA(n_components=4) 
pca.fit(dX)

print(pca.explained_variance_ratio_) 
print(pca.singular_values_)

pca = PCA(n_components=4, svd_solver='full')
pca.fit(dX)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)



### Q4.2 - Réaliser le diagnostic de variance avec un graphique à barre (barchart)


# ---------- Utiliser une librairie usuelle


#CODE
import numpy as np
from sklearn.decomposition import PCA

PCs=pca.explained_variance_ratio_
print(pca.singular_values_)
print(PCs)
import matplotlib.pyplot as plt
bars = ('PC1', 'PC2', 'PC3', 'PC4')
y_pos = np.arange(len(bars))
 
# Create bar plot
plt.bar(y_pos,PCs)
plt.xticks(y_pos, bars)
plt.show()

"""
[0.6425408  0.17533291 0.12055986 0.06156644]
"""


### Q4.3 - Combien de composantes doit-on garder? Pourquoi?
       
"""
Nous proposons de garder 2 composantes principales.
Avec les 2 premières nous explicons 81.5% de la variabilité.
Ceci permet de reduire la dimensionalité à la moyenne et avec +80% de variance.
"""



### Q4.4 - Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premières CP
###        Sélectionner éventuellement un sous-échantillon de points pour faciliter la visualisation

 

# ---------- Utiliser une librairie usuelle


#CODE
xvector = pca.components_[0]
yvector = pca.components_[1]

xs = pca.transform(X_scaled) [:,0]
ys = pca.transform(X_scaled)[:,1]

points_plot_index = np.random.randint(0, len(xs), 1000)

for i in points_plot_index :
    plt.plot(xs[i],ys[i],'bo')
    #plt.text(xs[i]*1.2, ys[i]*1.2, list(UsualData_clean[input_var].index)[i], color='b')

for i in range(len(xvector)):
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,
             list(UsualData_clean[input_var].columns.values)[i], color='r')
plt.show()

### Q4.5 - Comment les variables initiales se situent-elles par rapport aux 2 premières CP? 

"""
Les variables initialles de longitude se situent vers le côté negatif de la PC1 et positives par rapport à la PC2.
Les variables de latitude sont croissantes par rapport aux deux premières PCs.
"""





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
print('R2 regression lineaire usual data : %.2f' % r2)
#R²=0.0001252992167616318

print('Intercept regression lineaire usual data : %.2f' % regr.intercept_)
#intercept=9.760390606573518e-16
print('Coefficients regression lineaire usual data : %.2f' % regr.coef_)
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

import dask_ml
#import dask
from dask_glm.estimators import LinearRegression

#dask_ml.linear_model.LinearRegression

lr=LinearRegression()
lr.fit(X_scaled_bd,y_scaled_bd)
#X_scaled: matrice d entree transformee, y_scaled: matrice de sortie transformee
#Prediction modele Big Data
prediction_biglm=lr.predict(X_scaled_bd)



#Visualisation
plt.scatter(y_scaled_bd,prediction_biglm,color='black')
#y_scaled: valeur reelle, prediction_biglm: valeur predite
plt.xticks(())
plt.yticks(())
plt.show()

#Diagnostic des modeles
prediction_biglm=lr.predict(X_scaled_bd)
#on applique le score sur entree X_scaled et sortie y_scaled
lr.score(X_scaled_bd,y_scaled_bd)

#on imprime les valeurs des coefficients qui sont stockes dans l objet
#regression lineaire
print('Coefficients regression lineaire big data:\n ',lr.coef_)

#evaluation de l erreur moyenne de prediction par rapport a la veritable valeur
print("Erreur (RMS) regression lineaire big data: %.2f"
      %mean_squared_error(y_scaled_bd,prediction_biglm))

print('Variance score regression lineaire bigdata : %.2f' % r2_score(y_scaled_bd,prediction_biglm))





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

idx_train=np.random.rand(len(y_scaled))<0.8
#on cree jeu entrainement et test
X_apprentissage,X_intermediaire= X_scaled[idx_train],X_scaled[~idx_train]
y_apprentissage,y_intermediaire=y_scaled[idx_train],y_scaled[~idx_train]


idx_train_bis=np.random.rand(len(y_intermediaire))<0.5
X_validation,X_test= X_intermediaire[idx_train_bis],X_intermediaire[~idx_train_bis]
y_validation,y_test=y_intermediaire[idx_train_bis],y_intermediaire[~idx_train_bis]


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Réaliser la régression linéaire sur l'échantillon d'apprentissage, tester plusieurs valeurs
# de régularisation (hyperparamètre de la régression linéaire) et la qualité de prédiction sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

regr_app= linear_model.LinearRegression()

regr_app.fit(X_apprentissage,y_apprentissage)

# Evaluation du modele
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
y_predict_app = regr.predict(X_apprentissage)

#Calcul de l erreur quadratique moyenne (rmse)
rmse = (np.sqrt(mean_squared_error(y_apprentissage,y_predict_app)))
print(rmse)
#rmse=1.0005020337805912

#calcul du R²
r2=r2_score(y_apprentissage,y_predict_app)
print(r2)
#R²=0.00012750782767367852

print(regr_app.intercept_)
#intercept=0.00012738640200585854
print(regr_app.coef_)
#coefficients des variables explicatives "pickup_longitude","pickup_latitude",
#dropoff_longitude", "dropoff_latitude"
#[ 0.00347971 -0.00166427  0.00561088 -0.00302047]


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE

# Calculer le RMSE et le R² sur le jeu de test.



# ---------- Utiliser une librairie usuelle

regr_test= linear_model.LinearRegression()

regr_test.fit(X_test,y_test)

# Evaluation du modele
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
y_predict_test = regr.predict(X_test)

#Calcul de l erreur quadratique moyenne (rmse)
rmse = (np.sqrt(mean_squared_error(y_test,y_predict_test)))
print(rmse)
#rmse=0.9982937218934892

#calcul du R²
r2=r2_score(y_test,y_predict_test)
print(r2)
#R²=0.00014069263284666178




# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE

# Quelle est la qualité de la prédiction sur le jeu de test ?

#Le coefficient de détermination R² est faible, à 0.00014, donc la qualite de prediction est faible.
#On peut observer le graphique de la variable y prévue par le modele par rapport à la variable y reelle,
#sur le jeu de test,avec le code suivant:

#Visualisation
plt.scatter(y_test,y_predict_test,color='black')
#y_scaled: valeur reelle, prediction_biglm: valeur predite
plt.xticks(())
plt.yticks(())
plt.show()


#
# QUESTION 6 - REGRESSION LOGISTIQUE
# 



### Q6.1 - Mener une régression logisitique de la sortie "fare_amount" (après binarisation selon la médiane) 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# Créer la sortie binaire 'fare_binaire' en prenant la valeur médiane de "fare_amount" comme seuil


# ---------- Utiliser une librairie usuelle

#CODE
from sklearn.preprocessing import StandardScaler
X, y = UsualData_clean[input_var], UsualData_clean["fare_amount"] 
X_scaled - StandardScaler().fit_transform(X)

y.plot.hist
y.mean() 
y.median()

"""
moyenne 11.337383249349633
mediane 8.5
"""

y_binaire = np.zeros(len(y))
y_binaire[y>y.median()]= 1



# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Mener la régression logistique de "fare_binaire" en fonction des entrées standardisées


# ---------- Utiliser une librairie usuelle

#CODE
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression() 
log_reg.fit(X_scaled, y_binaire)

prediction_logreg = log_reg.predict(X_scaled) 
pred_proba_logreg = log_reg.predict_proba(X_scaled)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE




### Q6.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?

# Les coefficients
from sklearn.linear_model import LogisticRegression
print('Coefficients: \n', log_reg.coef_) 

"""
'pickup_longitude' 'pickup_latitude' 'dropoff_longitude' 'dropoff_latitude'
Coefficients: 
 [[ 0.00244204 -0.00291349 -0.00429839  0.0008465 ]]
 
Toutes les variables du modèle sont significatives
Plus les voyages sont long nord-sud plus le tariff sera grande.
Dans le cas des voyages est-ouest, plus on demarre à l'ouest plus la tariffe augmentera. 
 
"""

### Q6.3 - Prédire la probabilité que la course soit plus élevée que la médiane
#           en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

#CODE
idx_train = np.random.rand(len(y_binaire)) < 0.6
Xtrain, Xtest2 = X_scaled[idx_train], X_scaled[~idx_train]
ytrain, ytest2 = y_binaire[idx_train], y_binaire[~idx_train]

idx_train2 = np.random.rand(len(ytest2)) < 0.5
Xtest, Xvalidation = Xtest2[idx_train2], Xtest2[~idx_train2]
ytest, yvalidation = ytest2[idx_train2], ytest2[~idx_train2]


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Réaliser la régression logistique sur l'échantillon d'apprentissage et en testant plusieurs valeurs
# de régularisation (hyperparamètre de la régression logistique) sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

#CODE
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression() 
log_reg.fit(Xvalidation, yvalidation)

prediction_Xvalidation = log_reg.predict(Xvalidation) 
pred_proba_Xvalidation = log_reg.predict_proba(Xvalidation)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Calculer la précision (accuracy) et l'AUC de la prédiction sur le jeu de test.


# ---------- Utiliser une librairie usuelle

#CODE
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression() 
log_reg.fit(Xtest, ytest)

prediction_Xtest = log_reg.predict(Xtest) 
pred_proba_Xtest = log_reg.predict_proba(Xtest)

from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(ytest, log_reg.predict(Xtest))
confusion_mat * 100 / sum(sum(confusion_mat))

"""
array([[5.27768129e+01, 3.60900410e-04],
       [4.72224653e+01, 3.60900410e-04]])
"""

from sklearn import metrics
fpr, tpr, seuils = metrics.roc_curve (ytest, pred_proba_Xtest[:,1])
roc_auc = metrics.auc (fpr, tpr)
plt.plot(fpr, tpr, lw=1, alpha=0.3, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
metrics.accuracy_score(ytest, prediction_Xtest)
metrics.roc_auc_score(ytest, pred_proba_Xtest[:,1])

"""
Accuracy: 0.5277717377084087
AUC: 0.4931444903340024    
    
"""

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Quelle est la qualité de la prédiction sur le jeu de test ?

"""
La qualité de la prédiction du jeu de données est moyenne.
Avec une accuracy de 52% et une AUC de 49% nous considerons qu'il y a d'autres variables qui affectent le tariff.                    
"""












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

