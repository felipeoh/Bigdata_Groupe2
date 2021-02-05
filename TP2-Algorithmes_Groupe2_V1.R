# -*- coding: utf-8 -*-
'''
Created on Wed Feb 13 19:38:53 2019

@author:Groupe 2
BAKOP-KAMDEM Armel
LAFAY Anne
OROZCO-HERNANDEZ Felipe

'''



##############################################################################
#
#    ALGORITHMIE DU BIG DATA
#
##############################################################################


#
# QUESTION 0 - IMPORTATION DES PACKAGES ET LIBRAIRIES UTILISEES PAR LA SUITE
# 



CODE
library(bigmemory)
library(biganalytics)
library(glmnet)
library(ROCR)

#
# QUESTION 1 - IMPORT DU JEU DE DONNEES
# 


### Q1.1 - Indiquer le dossier et le fichier cible


CODE
setwd("D:/Mis Documentos/Formation Continue/4. Estadística/Toulouse Master/5. Cours M2 Stats/5. T2 2021/Big data/2. Les algorithmes du Big Data/Travaux pratiques")
fichier.Usualdata<-"train_echantillon.csv"
fichier.BDdata<-"train.csv"

### Q1.2 - Importer les jeux de données complets et échantillonnés
###        Prediction du prix du taxi à New York - https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data


# ---------- Utiliser une librairie usuelle (version de fichier échantillonnée)

#CODE

Usualdata<-read.csv(fichier.Usualdata,sep=",")
str(Usualdata)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory) (version complète du fichier)

CODE

BDdata<- read.big.matrix(fichier.BDdata,sep = ",",header = TRUE) 
str(BDdata)


#
# QUESTION 2 - PREPARATION/NETTOYAGE DU JEU DE DONNEES
# 


### Q2.1 - Nettoyer et préparer les données


# Enlever les valeurs incorrectes ou manquantes (si pertinent)


# ---------- Utiliser une librairie usuelle

CODE
#2.1.1. Usualdata
#----
# 5542385 cas
attach(Usualdata)

#Exploration de cas manquantes
sapply(Usualdata, function(x) sum(is.na(x)))
#-->Les variables dropoff on des NA, nous allons les enlever:

Usualdata_clean<-Usualdata[!is.na(dropoff_longitude),]
sapply(Usualdata_clean, function(x) sum(is.na(x)))
#--> tous les cas sont !NA 
# 5542347 cas qui restent
#----


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE
BDdata_clean<-na.omit(BDdata)


# Ne garder que les variables de géolocalisation (pour le jeu de données en entrée) et
# la variable "fare_amount" pour la sortie


# ---------- Utiliser une librairie usuelle

CODE
attach(Usualdata_clean)
Usualdata_clean<-Usualdata_clean[,c( "fare_amount","pickup_longitude","pickup_latitude",
                               "dropoff_longitude", "dropoff_latitude")]

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE

#****************#
# TO BE DONE
#****************#

#BDdata_clean<-BDdata_clean[,c( "fare_amount","pickup_longitude","pickup_latitude",
#                               "dropoff_longitude", "dropoff_latitude")]

#BDdata_clean<-as.big.matrix(BDdata_clean)


# Obtenir les caractéristiques statistiques de base des variables d'entrée et de sortie
# (par exemple, min, moyenne, mdéiane, max) et filter les valeurs aberrantes


# ---------- Utiliser une librairie usuelle

CODE
attach(Usualdata_clean)
summary(Usualdata_clean)
sapply(Usualdata_clean, function(x) sum(is.na(x)))

Usualdata_clean<-Usualdata_clean[Usualdata_clean$fare_amount>=0,]

#Usualdata_clean<-Usualdata_clean[pickup_longitude>=-76 & pickup_longitude<=-70,]
Usualdata_clean<-Usualdata_clean[Usualdata_clean$pickup_longitude>=-76,]
Usualdata_clean<-Usualdata_clean[Usualdata_clean$pickup_longitude<=-70,]

#Usualdata_clean<-Usualdata_clean[dropoff_longitude>=-76 & dropoff_longitude<=-70,]
Usualdata_clean<-Usualdata_clean[Usualdata_clean$dropoff_longitude>=-76,]
Usualdata_clean<-Usualdata_clean[Usualdata_clean$dropoff_longitude<=-70,]

#Usualdata_clean<-Usualdata_clean[pickup_latitude>=40 & pickup_latitude <=45,]
Usualdata_clean<-Usualdata_clean[Usualdata_clean$pickup_latitude>=40,]
Usualdata_clean<-Usualdata_clean[Usualdata_clean$pickup_latitude<=45,]

#Usualdata_clean<-Usualdata_clean[dropoff_latitude>=40 & dropoff_latitude<=45,]
Usualdata_clean<-Usualdata_clean[Usualdata_clean$dropoff_latitude>=40,]
Usualdata_clean<-Usualdata_clean[Usualdata_clean$dropoff_latitude<=45,]

attach(Usualdata_clean)
summary(Usualdata_clean)
sapply(Usualdata_clean, function(x) sum(is.na(x)))

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)


library(biganalytics)
summary(BDdata_clean)
"""
summary(BDdata_clean)
                          min         max        mean         NAs
fare_amount        -300.00000 93963.36000    11.34505     0.00000
pickup_longitude  -3442.05957  3457.62568   -72.50968     0.00000
pickup_latitude   -3492.26377  3408.78957    39.91979     0.00000
dropoff_longitude -3442.02457  3457.62235   -72.51121   376.00000
dropoff_latitude  -3547.88670  3537.13253    39.92068   376.00000
"""
#****************#
# A COMPLETER POUR LE FILTRAGE DES VALEURS ABERRANTES
#****************#


# Visualiser les distributions des variables d'entrée et de sortie (histogUsualme, pairplot)


# ---------- Utiliser une librairie usuelle

#histogramme de la variable a expliquer fare_amount
hist(Usualdata_clean[,"fare_amount"],main="fare_amount")

#histogramme des variables explicatives
hist(Usualdata_clean[,"pickup_longitude"],main="pickup_longitude")
hist(Usualdata_clean[,"pickup_latitude"],main="pickup_latitude")
hist(Usualdata_clean[,"dropoff_longitude"],main="dropoff_longitude")
hist(Usualdata_clean[,"dropoff_latitude"],main="dropoff_latitude")

#****************#
# LE CODE SUIVANT FAIT PLANTER R: VOIR COMMENT FAIRE POUR LES PAIRPLOTS
#****************#
       
#pairplots:
#graphiques du coût des taxis (fare_amount) en fonction de chaque variable
plot(Usualdata_clean[,"fare_amount"],Usualdata_clean[,"pickup_longitude"],pch=19,cex=0.8)
plot(Usualdata_clean[,"fare_amount"],Usualdata_clean[,"pickup_latitude"],pch=19,cex=0.8)
plot(Usualdata_clean[,"fare_amount"],Usualdata_clean[,"dropoff_longitude"],pch=19,cex=0.8)
plot(Usualdata_clean[,"fare_amount"],Usualdata_clean[,"dropoff_latitude"],pch=19,cex=0.8)

variables<-c("fare_amount","pickup_longitude","pickup_latitude",
             "dropoff_longitude", "dropoff_latitude")
#ensemble des paires de scatter plots avec la fonction pairs
pairs(Usualdata_clean[,variables])       
       
       
       
# Séparer la variable à prédire ("fare_amount") des autres variables d'entrée
# Créer un objet avec variables d'entrée et un objet avec valeurs de sortie (i.e. "fare_amount")



# ---------- Utiliser une librairie usuelle

CODE

x_input=Usualdata_clean[,c("pickup_longitude","pickup_latitude",
                           "dropoff_longitude", "dropoff_latitude")]
y_output=Usualdata_clean[,"fare_amount"]


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)


CODE

#****************#
# TO BE DONE
#****************#

# Standardiser la matrice d'entrée et les vecteurs de sortie (créer un nouvel objet)


# ---------- Utiliser une librairie usuelle

CODE

x_scale<-as.data.frame(scale(x_input))
y_scale<-as.data.frame(scale(y_output))
colnames(y_scale)=("fare_amount")

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE



#
# QUESTION 3 - CLUSTERING DU JEU DE DONNEES
# 




### Q3.1 - Réaliser un clustering k-means sur les données d'entrée standardisées


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE



### Q3.2 - Tracer la figure de l'inertie intraclusters et du R² en fonction du nombre de  clusters


# ---------- Utiliser une librairie usuelle

CODE





### Q3.3 - A partir de combien de clusters on peut dire que partitionner n'apporte plus 
###        grand chose? Pourquoi?



REPONSE ECRITE (3 lignes maximum)





### Q3.4 - Comment pouvez-vous qualifier les clusters obtenus selon les variables originales?
###        Par exemple, y a-t-il des clusters selon la localisation ? 




REPONSE ECRITE (3 lignes maximum)



### Q3.5 - Visualiser les clusters avec des couleurs différentes sur un 'pairplot' avec plusieurs variables


# ---------- Utiliser une librairie usuelle


CODE









#
# QUESTION 4 - ANALYSE EN COMPOSANTES PRINCIPALES (ACP) POUR SIMPLIFIER LE JEU DE DONNEES
# 



### Q4.1 - Faire une ACP sur le jeu de données standardisé


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


### Q4.2 - Réaliser le diagnostic de variance avec un graphique à barre (barchart)

 

# ---------- Utiliser une librairie usuelle


CODE




### Q4.3 - Combien de composantes doit-on garder? Pourquoi?
       


REPONSE ECRITE (3 lignes maximum)




### Q4.4 - Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premières CP
###        Sélectionner éventuellement un sous-échantillon de points pour faciliter la visualisation

 

# ---------- Utiliser une librairie usuelle


CODE




### Q4.5 - Comment les variables initiales se situent-elles par rapport aux 2 premières CP? 


REPONSE ECRITE (3 lignes maximum)











#
# QUESTION 5 - REGRESSION LINEAIRE
# 



### Q5.1 - Mener une régression linéaire de la sortie "fare_amount" 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# ---------- Utiliser une librairie usuelle

formule_modele<-as.formula(fare_amount~.)
#lm: fonction de regression lineaire
modele_lm<-lm(formule_modele,data=Usualdata_clean)
summary(modele_lm)

"""
Call:
lm(formula = formule_modele, data = Usualdata_clean)

Residuals:
    Min      1Q  Median      3Q     Max 
-413.28   -4.42   -1.87    2.00  488.02 

Coefficients:
                    Estimate Std. Error t value Pr(>|t|)    
(Intercept)        1.318e+04  1.093e+01  1206.1   <2e-16 ***
pickup_longitude   7.269e+01  9.832e-02   739.2   <2e-16 ***
pickup_latitude   -5.952e+01  1.376e-01  -432.5   <2e-16 ***
dropoff_longitude  4.774e+01  1.035e-01   461.3   <2e-16 ***
dropoff_latitude  -4.494e+01  1.267e-01  -354.7   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 8.556 on 5425725 degrees of freedom
Multiple R-squared:  0.2189,	Adjusted R-squared:  0.2189 
F-statistic: 3.801e+05 on 4 and 5425725 DF,  p-value: < 2.2e-16
"""
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE
#****************#
# TO BE DONE
#****************#

### Q5.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?


#Toutes les variables explicatives sont significatives (pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude):
#elles ont toutes une pvalue <2e-16 donc inférieure à 5%.
#La plus fortement liée à la variable à estimer, fare_amount, est pickup_longitude. La longitude du début de la course du taxi a un impact important
#sur le prix de la course du taxi.



### Q5.3 - Prédire le prix de la course en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle
#on cree les 3 echantillons: 60% de la base usual clean pour l echantillon d apprentissage
#puis 50% des 40% restants pour validation et test, ce qui fait 20% et 20%
idx_train<-sample(round(nrow(Usualdata_clean)*0.6))
Usualdata_clean_apprentissage<-Usualdata_clean[idx_train,]
intermediaire<-Usualdata_clean[-idx_train,]
idx_train_bis<-sample(round(nrow(intermediaire)*0.5))
Usualdata_clean_validation<-intermediaire[idx_train_bis,]
Usualdata_clean_test<-intermediaire[-idx_train_bis,]


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE
#****************#
# TO BE DONE
#****************#
""" a faire
library(biglm)
formule_modele<-as.formula(paste0("fare_amount~",paste0(input_var,collapse = " + ")))
modele_biglm<-bigglm(formule_modele,data=donnes_reg)
summary(modele_biglm)
prediction_biglm<-predict(modele_biglm,donnees_reg)
predict_index<-as.integer(rownames(prediction_biglm))
prediction_biglm<-predict(modele_biglm,donnees_reg)
predict_index<-as.integer(rownames(prediction_biglm))
plot(y_output[predict_index],prediction_biglm,pch=19,cex=0.8)
summary(modele_biglm)
"""

# Réaliser la régression linéaire sur l'échantillon d'apprentissage, tester plusieurs valeurs
# de régularisation (hyperpaUsualètre de la régression linéaire) et la qualité de prédiction sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

formule_modele<-as.formula(fare_amount~.)
#lm: fonction de regression lineaire
modele_lm_Usualdata_clean_apprentissage<-lm(formule_modele,data=Usualdata_clean_apprentissage)
summary(modele_lm_Usualdata_clean_apprentissage)
"""

Call:
lm(formula = formule_modele, data = Usualdata_clean_apprentissage)

Residuals:
    Min      1Q  Median      3Q     Max 
-400.19   -4.43   -1.88    2.00  479.29 

Coefficients:
                    Estimate Std. Error t value Pr(>|t|)    
(Intercept)       13086.6371    14.0937   928.5   <2e-16 ***
pickup_longitude     72.5386     0.1269   571.6   <2e-16 ***
pickup_latitude     -59.9590     0.1774  -338.1   <2e-16 ***
dropoff_longitude    46.7456     0.1335   350.1   <2e-16 ***
dropoff_latitude    -44.3640     0.1631  -272.1   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 8.569 on 3255433 degrees of freedom
Multiple R-squared:  0.2171,	Adjusted R-squared:  0.2171 
F-statistic: 2.257e+05 on 4 and 3255433 DF,  p-value: < 2.2e-16
"""

#tester la qualité de prédiction sur l'échantillon de validation
x_input_validation=Usualdata_clean_validation[,c("pickup_longitude","pickup_latitude",
                           "dropoff_longitude", "dropoff_latitude")]
y_output_validation=Usualdata_clean_validation[,"fare_amount"]
x_scale_validation<-as.data.frame(scale(x_input_validation))
y_scale_validation<-as.data.frame(scale(y_output_validation))
colnames(y_scale_validation)=("fare_amount")

prediction_lm_Usualdata_clean_validation<-predict(modele_lm_Usualdata_clean_apprentissage,x_input_validation)
print(prediction_lm_Usualdata_clean_validation)
plot(y_output_validation,prediction_lm_Usualdata_clean_validation,pch=19,cex=0.8)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

# Calculer le RMSE et le R² sur le jeu de test.

# ---------- Utiliser une librairie usuelle

modele_lm_Usualdata_clean_test<-lm(formule_modele,data=Usualdata_clean_test)
summary(modele_lm_Usualdata_clean_test)
"""
Call:
lm(formula = formule_modele, data = Usualdata_clean_test)

Residuals:
    Min      1Q  Median      3Q     Max 
-274.40   -4.39   -1.85    2.01  490.69 

Coefficients:
                    Estimate Std. Error t value Pr(>|t|)    
(Intercept)       13415.7778    24.3983   549.9   <2e-16 ***
pickup_longitude     74.5661     0.2212   337.1   <2e-16 ***
pickup_latitude     -58.0679     0.3097  -187.5   <2e-16 ***
dropoff_longitude    49.9563     0.2310   216.3   <2e-16 ***
dropoff_latitude    -44.8229     0.2849  -157.3   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 8.517 on 1085141 degrees of freedom
Multiple R-squared:  0.225,	Adjusted R-squared:  0.225 
F-statistic: 7.874e+04 on 4 and 1085141 DF,  p-value: < 2.2e-16
"""

#Le R² est 0.225

#RMSE: on utilise la fonction rmse(actual, predicted)
x_input_test=Usualdata_clean_test[,c("pickup_longitude","pickup_latitude",
                                     "dropoff_longitude", "dropoff_latitude")]
y_output_test=Usualdata_clean_test[,"fare_amount"]
x_scale_test<-as.data.frame(scale(x_input_test))
y_scale_test<-as.data.frame(scale(y_output_test))
colnames(y_scale_test)=("fare_amount")

prediction_lm_Usualdata_clean_test<-predict(modele_lm_Usualdata_clean_apprentissage,x_input_test)
#install.packages("Metrics")
library(Metrics)
prediction_lm_Usualdata_clean_test<-as.data.frame(prediction_lm_Usualdata_clean_test)
rmse(y_scale_test,prediction_lm_Usualdata_clean_test)

       
#****************#
# A FAIRE: RMSE() RENVOIE une erreur: ne fonctionne pas non plus quand on transforme
#y_scale_test (valeur réelle de la variable de sortie) et prediction_lm_Usualdata_clean_test (valeur estimée de la variable de sortie)
# en numérique   
#****************#

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE
#****************#
# TO BE DONE
#****************#
       
# Quelle est la qualité de la prédiction sur le jeu de test ?

#Le coefficient de détermination R² est faible, à 0.225, donc la qualite de prediction est faible.
#On peut observer le graphique de la variable y prévue par le modele par rapport à la variable y reelle,
#sur le jeu de test,avec le code suivant:
plot(y_output_test,prediction_lm_Usualdata_clean_test,pch=19,cex=0.8)









#
# QUESTION 6 - REGRESSION LOGISTIQUE
# 



### Q6.1 - Mener une régression logisitique de la sortie "fare_amount" (après binarisation selon la médiane) 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# Créer la sortie binaire 'fare_binaire' en prenant la valeur médiane de "fare_amount" comme seuil


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Mener la régression logistique de "fare_binaire" en fonction des entrées standardisées


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE




### Q6.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?



REPONSE ECRITE (3 lignes maximum)



### Q6.3 - Prédire la probabilité que la course soit plus élevée que la médiane
#           en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Réaliser la régression logistique sur l'échantillon d'apprentissage et en testant plusieurs valeurs
# de régularisation (hyperpaUsualètre de la régression logistique) sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Calculer la précision (accuracy) et l'AUC de la prédiction sur le jeu de test.



# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Quelle est la qualité de la prédiction sur le jeu de test ?


REPONSE ECRITE (3 lignes maximum)







#
# QUESTION 7 - RESEAU DE NEURONES (QUESTION BONUS)
# 



### Q7.1 - Mener une régression de la sortie "fare_amount" en fonction de l'entrée (mise à l'échelle), 
###       sur tout le jeu de données, avec un réseau à 2 couches cachées de 10 neurones chacune



# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE



### Q7.2 - Prédire le prix de la course en fonction de nouvelles entrées avec le réseau de neurones entraîné


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Réaliser la régression avec réseau de neurones sur l'échantillon d'apprentissage et en testant plusieurs 
# nombre de couches et de neurones par couche sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Calculer le RMSE et le R² de la meilleure prédiction sur le jeu de test.


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE

# Quelle est la qualité de la prédiction sur le jeu de test ? Comment se compare-t-elle à la régression linéaire?


REPONSE ECRITE (3 lignes maximum)


