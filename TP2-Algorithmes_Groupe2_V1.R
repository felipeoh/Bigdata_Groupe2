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
library(biglm)

#
# QUESTION 1 - IMPORT DU JEU DE DONNEES
# 


### Q1.1 - Indiquer le dossier et le fichier cible


CODE
setwd("D:/Mis Documentos/Formation Continue/4. Estadística/Toulouse Master/5. Cours M2 Stats/5. T2 2021/Big data/2. Les algorithmes du Big Data/Travaux pratiques")
fichier.Usualdata<-"train_echantillon.csv"
fichier.BDdata<-"train_echantillon.csv"

# Le fichier train étant trop lourd pour l'ordinateur, nous allons utiliser le fichier train_echantillon à la fois pour la partie usuelle et 
#pour la partie libraire big data, en donnant des noms différents aux données et en réalisant des traitements différents

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

BDdata_clean<-BDdata_clean[,c( "fare_amount","pickup_longitude","pickup_latitude",
                               "dropoff_longitude", "dropoff_latitude")]

BDdata_clean<-as.big.matrix(BDdata_clean)


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

all_cols1 <- colnames(BDdata_clean)
keep_cols1 <- all_cols[!(all_cols %in% c("fare_amount"))]

all_cols2 <- colnames(BDdata_clean)
keep_cols2 <- all_cols[!(all_cols %in% c("pickup_longitude","pickup_latitude",
                                        "dropoff_longitude", "dropoff_latitude"))]

x_input_bd<-BDdata_clean[,keep_cols1]
x_input_bd<-as.big.matrix(x_input_bd)
y_output_bd<-BDdata_clean[,keep_cols2]

# Standardiser la matrice d'entrée et les vecteurs de sortie (créer un nouvel objet)


# ---------- Utiliser une librairie usuelle

CODE

x_scale<-as.data.frame(scale(x_input))
y_scale<-as.data.frame(scale(y_output))
colnames(y_scale)=("fare_amount")

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE
#Après avoir essayé plusieurs méthodes pour traiter les big matrix et n'ayant pas étudié ce type d'objets en cours,
#nous procedons à traiter la big.matrix comme un df

x_input_bd<-x_input_bd[]
x_input_bd<-as.data.frame(x_input_bd)

x_scale_bd<-as.data.frame(scale(x_input_bd))
y_scale_bd<-as.data.frame(scale(y_output_bd))
colnames(y_scale_bd)=("fare_amount")



#
# QUESTION 3 - CLUSTERING DU JEU DE DONNEES
# 




# ---------- Utiliser une librairie usuelle
#centers=nombre de clusters
#iter.max=nombre maximum d iterations

set.seed(20)
kmeans_clusters<-kmeans(x_input,centers=6,iter.max=100,algorithm = "Lloyd")
str(kmeans_clusters)

"""
List of 9
 $ cluster     : Named int [1:5425730] 4 1 2 5 2 3 1 1 2 5 ...
  ..- attr(*, "names")= chr [1:5425730] "2" "3" "4" "5" ...
 $ centers     : num [1:6, 1:4] -74 -74 -74 -74 -74 ...
  ..- attr(*, "dimnames")=List of 2
  .. ..$ : chr [1:6] "1" "2" "3" "4" ...
  .. ..$ : chr [1:4] "pickup_longitude" "pickup_latitude" "dropoff_longitude" "dropoff_latitude"
 $ totss       : num 29560 : variance totale
 $ withinss    : num [1:6] 1933 3056 986 1273 1372 ...: variance interne a chaque cluster
 $ tot.withinss: num 15452 : distance totale des points a leur barycentre
 $ betweenss   : num 14108 : distance entre les points et les barycentres
 $ size        : int [1:6] 1107233 1121105 1489870 153952 1355942 197628 : nombre de points qu on a dans chaque cluster
 $ iter        : int 101 : nombre d iterations necessaires pour la convergence
 $ ifault      : int 2
 - attr(*, "class")= chr "kmeans"
 """
for(cluster_num in 1:10){
  kmeans_clusters<-kmeans(x_scale,centers=cluster_num,iter.max = 200,algorithm = "Lloyd")
  print(paste0(cluster_num,
               "clusters - Inertie: ",
               kmeans_clusters$tot.withinss))
}

"""
[1] "1clusters - Inertie: 21702916.0000079"
[1] "2clusters - Inertie: 17674348.9938002"
[1] "3clusters - Inertie: 14506022.126452"
[1] "4clusters - Inertie: 13421219.1736507"
[1] "5clusters - Inertie: 12753739.9699624"
[1] "6clusters - Inertie: 9365473.95103332"
[1] "7clusters - Inertie: 10592348.6476492"
[1] "8clusters - Inertie: 10347720.5847138"
[1] "9clusters - Inertie: 7702293.68061266"
[1] "10clusters - Inertie: 9326962.04724501"
"""

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#library(biganalytics)
#kmeans_clusters<-bigkmeans(x_scale,centers=6,iter.max=100,nstart=1,dist="euclid")


### Q3.2 - Tracer la figure de l'inertie intraclusters et du R² en fonction du nombre de  clusters


# ---------- Utiliser une librairie usuelle

             

### Q3.3 - A partir de combien de clusters on peut dire que partitionner n'apporte plus 
###        grand chose? Pourquoi?



#Ici il n'y a pas de convergence. Donc l'inertie commence par diminuer, ce qui pourrait nous conduire a choisir 
#3 ou 4 clusters. Mais ensuite elle réaugmente.





### Q3.4 - Comment pouvez-vous qualifier les clusters obtenus selon les variables originales?
###        Par exemple, y a-t-il des clusters selon la localisation ? 




REPONSE ECRITE (3 lignes maximum)



### Q3.5 - Visualiser les clusters avec des couleurs différentes sur un 'pairplot' avec plusieurs variables


# ---------- Utiliser une librairie usuelle


index_plot<-sample(nrow(x_scale),1000)

pairs(x_scale[index_plot,],col=kmeans_clusters$cluster[index_plot],pch=19)









#
# QUESTION 4 - ANALYSE EN COMPOSANTES PRINCIPALES (ACP) POUR SIMPLIFIER LE JEU DE DONNEES
# 



### Q4.1 - Faire une ACP sur le jeu de données standardisé


# ---------- Utiliser une librairie usuelle

CODE
summary(x_input)

"""
pickup_longitude pickup_latitude dropoff_longitude dropoff_latitude
 Min.   :-75.97   Min.   :40.01   Min.   :-76.00    Min.   :40.01   
 1st Qu.:-73.99   1st Qu.:40.74   1st Qu.:-73.99    1st Qu.:40.74   
 Median :-73.98   Median :40.75   Median :-73.98    Median :40.75   
 Mean   :-73.98   Mean   :40.75   Mean   :-73.97    Mean   :40.75   
 3rd Qu.:-73.97   3rd Qu.:40.77   3rd Qu.:-73.97    3rd Qu.:40.77   
 Max.   :-70.00   Max.   :44.55   Max.   :-70.00    Max.   :44.72  

"""

ACP_transform<-prcomp(x_scale,
                      center = TRUE,
                      scale. = TRUE)
print(ACP_transform)

"""
Standard deviations (1, .., p=4):
  [1] 1.3532537 1.0486154 0.7598965 0.7011902

Rotation (n x k) = (4 x 4):
  PC1        PC2        PC3        PC4
pickup_longitude  0.4562510 -0.5673123  0.6278460 -0.2753201
pickup_latitude   0.5033922  0.5096504  0.3574696  0.5992231
dropoff_longitude 0.5044226 -0.4676294 -0.6345247  0.3525039
dropoff_latitude  0.5329064  0.4469183 -0.2745965 -0.6639816

"""

summary(ACP_transform)

"""
Importance of components:
                          PC1    PC2    PC3    PC4
Standard deviation     1.3533 1.0486 0.7599 0.7012
Proportion of Variance 0.4578 0.2749 0.1444 0.1229
Cumulative Proportion  0.4578 0.7327 0.8771 1.0000

"""

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE
x_scale_bd<-na.omit(x_scale_bd)
pca <- prcomp(x_scale_bd, center = TRUE, scale. = TRUE)

summary(pca)
"""
Importance of components:
                          PC1    PC2    PC3     PC4
Standard deviation     1.6032 0.8375 0.6944 0.49625
Proportion of Variance 0.6425 0.1753 0.1206 0.06157
Cumulative Proportion  0.6425 0.8179 0.9384 1.00000
"""

### Q4.2 - Réaliser le diagnostic de variance avec un graphique à barre (barchart)

 

# ---------- Utiliser une librairie usuelle


CODE

plot(ACP_transform, xlab="Composante",main="ACP Coordonnées")


### Q4.3 - Combien de composantes doit-on garder? Pourquoi?
       
"""
Avec les deux premières composantes principales la variabilité est expliquée à 74%.
Vu que nous avons 4 variables et une bonne explication avec 2, nous achevons une réduction correcte de dimensionalité.
Nous proposons de garder les 2 premières composantes.

"""


### Q4.4 - Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premières CP
###        Sélectionner éventuellement un sous-échantillon de points pour faciliter la visualisation

 

# ---------- Utiliser une librairie usuelle


CODE

taxi.pca <- ACP_transform
taxi.pca$x<-taxi.pca$x[sample(nrow(taxi.pca$x), 1000), ]

biplot(taxi.pca)

### Q4.5 - Comment les variables initiales se situent-elles par rapport aux 2 premières CP? 


"""
Les variables de latitude augementent par rapport aux PC1 et PC2
Les variables de longitude augementent par rapport à la PC1 et diminuent par à la PC2.

"""











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

y_binaire<-rep(0,nrow(Usualdata_clean))
y_binaire[y_output>median(y_output)]<-1

Usualdata_clean2<-x_scale
Usualdata_clean2$fare_amount<-y_binaire
#table(Usualdata_clean2$fare_amount)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE
y_binaire_bd<-rep(0,nrow(BDdata_clean3))
y_binaire_bd[y_output_bd>median(y_output_bd)]<-1

BDdata_clean2<-x_scale_bd
BDdata_clean2$fare_amount<-y_binaire_bd


# Mener la régression logistique de "fare_binaire" en fonction des entrées standardisées


# ---------- Utiliser une librairie usuelle

CODE

modele_logit<-glm(fare_amount~pickup_longitude+pickup_latitude+dropoff_longitude+dropoff_latitude,
                  family = binomial(link = "logit"),
                  data = Usualdata_clean2)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE
library(biglm)

modele_logit_big<- bigglm(fare_amount~pickup_longitude+ pickup_latitude + dropoff_longitude+ dropoff_latitude,
                       family=binomial(link=logit), data = BDdata_clean2, chunksize=1000, maxit=10)


### Q6.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?

summary(modele_logit)

"""
glm(formula = fare_amount ~ pickup_longitude + pickup_latitude + 
    dropoff_longitude + dropoff_latitude, family = binomial(link = "logit"), 
    data = Usualdata_clean2)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-8.4904  -1.0912  -0.9395   1.2434   8.4904  

Coefficients:
                    Estimate Std. Error z value Pr(>|z|)    
(Intercept)       -0.0882813  0.0008915  -99.03   <2e-16 ***
pickup_longitude   0.2845351  0.0013766  206.69   <2e-16 ***
pickup_latitude   -0.2202096  0.0012347 -178.36   <2e-16 ***
dropoff_longitude  0.4218159  0.0014539  290.13   <2e-16 ***
dropoff_latitude  -0.2508010  0.0012187 -205.80   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 7504422  on 5425729  degrees of freedom
Residual deviance: 7219606  on 5425725  degrees of freedom
AIC: 7219616

summary(modele_logit_big)

Large data regression model: bigglm(fare_amount ~ pickup_longitude + pickup_latitude + dropoff_longitude + 
    dropoff_latitude, family = binomial(link = logit), data = BDdata_clean2, 
    chunksize = 1000, maxit = 10)
Sample size =  5542347 
                     Coef    (95%     CI)     SE      p
(Intercept)       -0.1131 -0.1148 -0.1114 0.0009 0.0000
pickup_longitude   0.0025 -0.0004  0.0053 0.0014 0.0828
pickup_latitude   -0.0029 -0.0053 -0.0005 0.0012 0.0158
dropoff_longitude -0.0043 -0.0073 -0.0014 0.0015 0.0034
dropoff_latitude   0.0008 -0.0013  0.0030 0.0011 0.4400

Nous observons que les 4 variables son significatives pour le modèle normal.
Dans le cas du modèle big data, la variable dropoff_latitude n'est pas significative
Quand le lieu de départ est plus à l'est la probabilité d'avoir une tariffe élévé (long/large voyage) sera plus haute.
Quand le lieu d'arrivé est plus à l'est la probabilité d'avoir une tariffe élévé (long/large voyage) sera plus haute.
Le cas contraire s'observe pour les voyages sud-nord ou nord-sud.

"""

### Q6.3 - Prédire la probabilité que la course soit plus élevée que la médiane
#           en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

CODE
set.seed(20212)
idx_train<-sample(round(nrow(Usualdata_clean2)*0.6))
donnees_train<-Usualdata_clean2[idx_train,]
donnees_test <-Usualdata_clean2[-idx_train,]

idx_test<-sample(round(nrow(donnees_test)*0.5))
donnees_test2 <-donnees_test[idx_test,]
donnees_validation <-donnees_test[-idx_test,]
donnees_test<-donnees_test2
rm(donnees_test2)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE
set.seed(20212)
idx_train_bd<-sample(round(nrow(BDdata_clean2)*0.6))
donnees_train_bd<-BDdata_clean2[idx_train_bd,]
donnees_test_bd2 <-BDdata_clean2[-idx_train_bd,]

idx_test_bd<-sample(round(nrow(donnees_test_bd2)*0.5))
donnees_test_bd <-donnees_test_bd2[idx_test_bd,]
donnees_validation_bd <-donnees_test_bd2[-idx_test_bd,]
rm(donnees_test_bd2)


# Réaliser la régression logistique sur l'échantillon d'apprentissage et en testant plusieurs valeurs
# de régularisation (hyperpaUsualètre de la régression logistique) sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

CODE

modele_train<-glm(fare_amount~pickup_longitude+pickup_latitude+dropoff_longitude+dropoff_latitude,
                  family = binomial(link = "logit"),
                  data = donnees_train)


train_logit <-glmnet(as.matrix(x_scale),as.factor(y_binaire), family="binomial",alpha=0,standardize = FALSE)
train_logit$lambda
coef(train_logit)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE
modele_logit_big_train<- bigglm(fare_amount~pickup_longitude+ pickup_latitude + dropoff_longitude+ dropoff_latitude,
                          family=binomial(link=logit), data = donnees_train_bd, chunksize=1000, maxit=10)

summary(modele_logit_big_train)


# Calculer la précision (accuracy) et l'AUC de la prédiction sur le jeu de test.


# ---------- Utiliser une librairie usuelle

CODE
prediction_test <- predict(modele_train, donnees_test, type='response') 
hist (prediction_test)
prediction_test_binaire <- rep(0, length(prediction_test)) 
seuil_binaire <- 0.5 
prediction_test_binaire[prediction_test > seuil_binaire] <- 1 
matrice_confusion <- table (donnees_test$fare_amount ,prediction_test_binaire)
round (matrice_confusion * 100 / sum(matrice_confusion), 1)

"""
prediction_test_binaire
       0    1
  0 47.6  5.3
  1 33.1 14.0
"""
#Accuracy
Accuracy_test<-(matrice_confusion[1,1]+matrice_confusion[2,2])/sum(matrice_confusion)
Accuracy_test

"""
Accuracy_test
0.6157162
"""

#AUC
pr<-prediction(prediction_test,
               donnees_test$fare_amount)
auc<-performance(pr,measure="auc")
auc<-auc@y.values[[1]] 
auc

"""
AUC
0.5980835
"""


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE
       
prediction_test_bd <- predict (modele_logit_big_train, donnees_test_bd, type='response')
prediction_test_bd_binaire <- rep(0, length(prediction_test_bd))

seuil_binaire <- 0.5
prediction_test_bd_binaire[prediction_test_bd > seuil_binaire] <- 1

matrice_confusion_bd <- table (donnees_test_bd$fare_amount,
                            prediction_test_bd_binaire)
round(matrice_confusion_bd*100/sum(matrice_confusion_bd),1)


# Quelle est la qualité de la prédiction sur le jeu de test ?


"""
La qualité de la prédiction sur le jeu de teste est moyenne.
Avec un accuracy de 62% et un AUC de 60%.
Il existe d'autres variables qui doivent avoir un effet sur le tariff, par exemple: durée, type de tariff, etc.
"""







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


