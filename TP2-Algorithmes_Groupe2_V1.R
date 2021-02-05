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
summary(modele_lm)

Call:
lm(formula = formule_modele, data = Usualdata_clean)

Residuals:
   Min     1Q Median     3Q    Max 
-22.57  -5.32  -2.82   1.18 686.13 

Coefficients:
                    Estimate Std. Error t value Pr(>|t|)    
(Intercept)       12.0286087  0.0277514 433.442  < 2e-16 ***
pickup_longitude   0.0024522  0.0005259   4.663 3.12e-06 ***
pickup_latitude   -0.0012860  0.0005622  -2.288   0.0222 *  
dropoff_longitude  0.0046030  0.0005444   8.455  < 2e-16 ***
dropoff_latitude  -0.0031840  0.0005179  -6.148 7.83e-10 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 9.782 on 5116832 degrees of freedom
  (316140 observations deleted due to missingness)
Multiple R-squared:  0.0001243,	Adjusted R-squared:  0.0001236 
F-statistic: 159.1 on 4 and 5116832 DF,  p-value: < 2.2e-16
"""

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


### Q5.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?



REPONSE ECRITE (3 lignes maximum)



### Q5.3 - Prédire le prix de la course en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Réaliser la régression linéaire sur l'échantillon d'apprentissage, tester plusieurs valeurs
# de régularisation (hyperpaUsualètre de la régression linéaire) et la qualité de prédiction sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE

# Calculer le RMSE et le R² sur le jeu de test.



# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE

# Quelle est la qualité de la prédiction sur le jeu de test ?


REPONSE ECRITE (3 lignes maximum)








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


