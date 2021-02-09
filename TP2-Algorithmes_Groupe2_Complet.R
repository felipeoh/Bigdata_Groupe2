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
library(bigmemory)
library(biganalytics)
library(glmnet)
library(ROCR)
library(biglm)

library(neuralnet)
library(nnet)
library(keras)
#
# QUESTION 1 - IMPORT DU JEU DE DONNEES
# 
### Q1.1 - Indiquer le dossier et le fichier cible
dossier <- "C:/M2-Stat.Eco-Toulouse_UE/UE10-BIG DATA/Section 2-les algorithmes du Big Data/TP/data/"
csv_file <- "train_echantillon.csv"
bigcsv_file <- "train.csv"

chemin_db <- paste0(dossier,csv_file)
chemin_big_db <- paste0(dossier,bigcsv_file)


### Q1.2 - Importer les jeux de données complets et échantillonnÃ©s
###        Prediction du prix du taxi Ã  New York - https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data


# ---------- Utiliser une librairie usuelle (version de fichier Ã©chantillonnÃ©e)

Usualdata<-read.csv(chemin_db)
str(Usualdata)
nrow(Usualdata)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory) (version complÃ¨te du fichier)

BDdata<- read.big.matrix(chemin_big_db,sep = ",",header = TRUE) 
str(BDdata)


#
# QUESTION 2 - PREPARATION/NETTOYAGE DU JEU DE DONNEES
# 

### Q2.1 - Nettoyer et préparer les données

# Enlever les valeurs incorrectes ou manquantes (si pertinent)
# ---------- Utiliser une librairie usuelle
#2.1.1. Usualdata
#----
# 5542385 cas
attach(Usualdata)

#Exploration de cas manquantes
sapply(Usualdata, function(x) sum(is.na(x)))
#-->Les variables dropoff on des 38 NA, nous allons les enlever:

Usualdata_clean<-Usualdata[!is.na(dropoff_longitude),]
sapply(Usualdata_clean, function(x) sum(is.na(x)))
#--> tous les cas sont !NA 
# 5542347 cas qui restent
#----

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

BDdata_clean<-na.omit(BDdata)


# Ne garder que les variables de gÃ©olocalisation (pour le jeu de donnÃ©es en entrÃ©e) et
# la variable "fare_amount" pour la sortie


# ---------- Utiliser une librairie usuelle

attach(Usualdata_clean)
Usualdata_clean<-Usualdata_clean[,c( "fare_amount","pickup_longitude","pickup_latitude",
                               "dropoff_longitude", "dropoff_latitude")]

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

BDdata_clean<-BDdata_clean[,c( "fare_amount","pickup_longitude","pickup_latitude",
                               "dropoff_longitude", "dropoff_latitude")]

BDdata_clean<-as.big.matrix(BDdata_clean)

# Obtenir les caractéristiques statistiques de base des variables d'entrée et de sortie
# (par exemple, min, moyenne, médiane, max) et filter les valeurs aberrantes


# ---------- Utiliser une librairie usuelle

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

# échantillonnage (base apprentissage et base test)
# ---------- Utiliser une librairie usuelle
t <- nrow(Usualdata_clean)

set.seed(286745)
id_tir <- sample(1:t,floor(.7*t))

Usualdata_app <- Usualdata_clean[id_tir,]
Usualdata_test <- Usualdata_clean[-id_tir,]

# Visualiser les distributions des variables d'entrée et de sortie (histogUsualme, pairplot)


# ---------- Utiliser une librairie usuelle

#histogramme de la variable a expliquer fare_amount
hist(Usualdata_app[,"fare_amount"],main="fare_amount")

#histogramme des variables explicatives
hist(Usualdata_app[,"pickup_longitude"],main="pickup_longitude")
hist(Usualdata_app[,"pickup_latitude"],main="pickup_latitude")
hist(Usualdata_app[,"dropoff_longitude"],main="dropoff_longitude")
hist(Usualdata_app[,"dropoff_latitude"],main="dropoff_latitude")

#****************#
# LE CODE SUIVANT FAIT PLANTER R: VOIR COMMENT FAIRE POUR LES PAIRPLOTS
#****************#
       
#pairplots:
#graphiques du coût des taxis (fare_amount) en fonction de chaque variable
set.seed(286743)
id2 <- sample(1:nrow(Usualdata_app),10000) # extraction d'un échantillon plus réduit
Usualdata_reduit <- Usualdata_app[id2,]

plot(Usualdata_reduit[,"pickup_longitude"],Usualdata_reduit[,"fare_amount"],pch=19,cex=0.8)
plot(Usualdata_reduit[,"pickup_latitude"],Usualdata_reduit[,"fare_amount"],pch=19,cex=0.8)
plot(Usualdata_reduit[,"dropoff_longitude"],Usualdata_reduit[,"fare_amount"],pch=19,cex=0.8)
plot(Usualdata_reduit[,"dropoff_latitude"],Usualdata_reduit[,"fare_amount"],pch=19,cex=0.8)

variables<-c("fare_amount","pickup_longitude","pickup_latitude",
             "dropoff_longitude", "dropoff_latitude")
#ensemble des paires de scatter plots avec la fonction pairs
pairs(Usualdata_reduit[,variables])    

# corrélation de pearson
cor(Usualdata_reduit) # corrélation faible entre les variables
       
# Séparer la variable à prédire ("fare_amount") des autres variables d'entrée
# Créer un objet avec variables d'entrée et un objet avec valeurs de sortie (i.e. "fare_amount")

# ---------- Utiliser une librairie usuelle

x_input=Usualdata_reduit[,c("pickup_longitude","pickup_latitude",
                           "dropoff_longitude", "dropoff_latitude")]
y_output=Usualdata_reduit[,"fare_amount"]

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

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
x_scale<-as.data.frame(scale(x_input))
y_scale<-as.data.frame(scale(y_output))
colnames(y_scale)=("fare_amount")

Udata_test_sc <-as.data.frame(scale(Usualdata_test))

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
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
set.seed(19830)
kmeans_clusters<-kmeans(x_input,centers=6,iter.max=100,nstart=5,algorithm = "Lloyd")
print(kmeans_clusters)
str(kmeans_clusters)
"
List of 9
 $ cluster     : Named int [1:10000] 1 6 1 3 6 3 1 6 3 1 ...
  ..- attr(*, "names")= chr [1:10000] "4084456" "3964624" "216917" "3782023" ...
 $ centers     : num [1:6, 1:4] -74 -74 -74 -73.8 -73.9 ...
  ..- attr(*, "dimnames")=List of 2
  .. ..$ : chr [1:6] "1" "2" "3" "4" ...
  .. ..$ : chr [1:4] "pickup_longitude" "pickup_latitude" "dropoff_longitude" "dropoff_latitude"
 $ totss       : num 54.2
 $ withinss    : num [1:6] 6.3 2.6 3.11 7.75 1.71 ...
 $ tot.withinss: num 27.2
 $ betweenss   : num 27
 $ size        : int [1:6] 2537 283 4305 178 257 2440
 $ iter        : int 40
 $ ifault      : NULL
 - attr(*, "class")= chr "kmeans"
 "
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#library(biganalytics)
#kmeans_clusters<-bigkmeans(x_scale,centers=6,iter.max=100,nstart=1,dist="euclid")


### Q3.2 - Tracer la figure de l'inertie intraclusters et du R² en fonction du nombre de  clusters

# ---------- Utiliser une librairie usuelle
inertie.expl <- interie.intra <- rep(0,times=10)
for (k in 2:10){
  clus <- kmeans(x_scale,centers=k,nstart=5,iter.max = 200,algorithm = "Lloyd")
  interie.intra[k] <- clus$tot.withinss
  inertie.expl[k] <- clus$betweenss/clus$totss
}
plot(1:10,interie.intra,type="b",xlab="Nb. de clusters",ylab="Inertie intra-groupe")
plot(1:10,inertie.expl,type="b",xlab="Nb. de clusters",ylab="% Inertie expliquée (R²)")

### Q3.3 - A partir de combien de clusters on peut dire que partitionner n'apporte plus 
###        grand chose? Pourquoi?
"
Puisqu'il y a pas de convergence de l'inertie, nous allons observer d'autre de qualité à l instar 
de l'indice de Calinski Harabasz.
"
library(fpc)
sol.kmeans <- kmeansruns(x_scale,krange=2:10,criterion="ch",algorithm = "Lloyd")
plot(1:10,sol.kmeans$crit,type="b",xlab="Nb. de clusters",ylab="Calinski Harabasz")

"
à partir de cluster 3, l'augmentation de l'indice de calinski n'est plus significatif.
ainsi, le nombre de cluster nécessaire est de 3.
"
# classification finale

kmeans_clusters<-kmeans(x_input,centers=3,iter.max=100,nstart=5,algorithm = "Lloyd")

### Q3.4 - Comment pouvez-vous qualifier les clusters obtenus selon les variables originales?
###        Par exemple, y a-t-il des clusters selon la localisation ? 

plot(Usualdata_reduit$pickup_longitude,Usualdata_reduit$pickup_latitude,
                       col=kmeans_clusters$cluster,pch=19,cex=0.8)

"
Les clusters obtenus sont fonction des coordonnées géographiques.
Comme l'indique la figure ci-dessus. Elles dépendent de la latidue et de la longitude.

"

### Q3.5 - Visualiser les clusters avec des couleurs diffrentes sur un 'pairplot' avec plusieurs variables

# ---------- Utiliser une librairie usuelle

pairs(x_scale,col=kmeans_clusters$cluster,pch=16,cex=0.8)

#
# QUESTION 4 - ANALYSE EN COMPOSANTES PRINCIPALES (ACP) POUR SIMPLIFIER LE JEU DE DONNEES
# 

### Q4.1 - Faire une ACP sur le jeu de donnÃ©es standardisÃ©
# ---------- Utiliser une librairie usuelle
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

plot(ACP_transform, xlab="Composante",main="ACP CoordonnÃ©es")

### Q4.3 - Combien de composantes doit-on garder? Pourquoi?
       
"""
Avec les deux premiÃ¨res composantes principales la variabilitÃ© est expliquÃ©e Ã  74%.
Vu que nous avons 4 variables et une bonne explication avec 2, nous achevons une rÃ©duction correcte de dimensionalitÃ©.
Nous proposons de garder les 2 premiÃ¨res composantes.

"""


### Q4.4 - Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premiÃ¨res CP
###        SÃ©lectionner Ã©ventuellement un sous-Ã©chantillon de points pour faciliter la visualisation

 

# ---------- Utiliser une librairie usuelle


CODE

taxi.pca <- ACP_transform
taxi.pca$x<-taxi.pca$x[sample(nrow(taxi.pca$x), 1000), ]

biplot(taxi.pca)

### Q4.5 - Comment les variables initiales se situent-elles par rapport aux 2 premiÃ¨res CP? 


"""
Les variables de latitude augementent par rapport aux PC1 et PC2
Les variables de longitude augementent par rapport Ã  la PC1 et diminuent par Ã  la PC2.

""
#
# QUESTION 5 - REGRESSION LINEAIRE
# 
### Q5.1 - Mener une rÃ©gression linÃ©aire de la sortie "fare_amount" 
###        en fonction de l'entrÃ©e (mise Ã  l'Ã©chelle), sur tout le jeu de donnÃ©es


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
Signif. codes:  0 â€˜***â€™ 0.001 â€˜**â€™ 0.01 â€˜*â€™ 0.05 â€˜.â€™ 0.1 â€˜ â€™ 1

Residual standard error: 8.556 on 5425725 degrees of freedom
Multiple R-squared:  0.2189,	Adjusted R-squared:  0.2189 
F-statistic: 3.801e+05 on 4 and 5425725 DF,  p-value: < 2.2e-16
"""
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE
#****************#
# TO BE DONE
#****************#

### Q5.2 - Que pouvez-vous dire des rÃ©sultats du modÃ¨le? Quelles variables sont significatives?


#Toutes les variables explicatives sont significatives (pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude):
#elles ont toutes une pvalue <2e-16 donc infÃ©rieure Ã  5%.
#La plus fortement liÃ©e Ã  la variable Ã  estimer, fare_amount, est pickup_longitude. La longitude du dÃ©but de la course du taxi a un impact important
#sur le prix de la course du taxi.



### Q5.3 - PrÃ©dire le prix de la course en fonction de nouvelles entrÃ©es avec une rÃ©gression linÃ©aire


# Diviser le jeu de donnÃ©es initial en Ã©chantillons d'apprentissage (60% des donnÃ©es), validation (20%) et test (20%)


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

# RÃ©aliser la rÃ©gression linÃ©aire sur l'Ã©chantillon d'apprentissage, tester plusieurs valeurs
# de rÃ©gularisation (hyperpaUsualÃ¨tre de la rÃ©gression linÃ©aire) et la qualitÃ© de prÃ©diction sur l'Ã©chantillon de validation. 


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
Signif. codes:  0 â€˜***â€™ 0.001 â€˜**â€™ 0.01 â€˜*â€™ 0.05 â€˜.â€™ 0.1 â€˜ â€™ 1

Residual standard error: 8.569 on 3255433 degrees of freedom
Multiple R-squared:  0.2171,	Adjusted R-squared:  0.2171 
F-statistic: 2.257e+05 on 4 and 3255433 DF,  p-value: < 2.2e-16
"""

#tester la qualitÃ© de prÃ©diction sur l'Ã©chantillon de validation
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

# Calculer le RMSE et le RÂ² sur le jeu de test.

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
Signif. codes:  0 â€˜***â€™ 0.001 â€˜**â€™ 0.01 â€˜*â€™ 0.05 â€˜.â€™ 0.1 â€˜ â€™ 1

Residual standard error: 8.517 on 1085141 degrees of freedom
Multiple R-squared:  0.225,	Adjusted R-squared:  0.225 
F-statistic: 7.874e+04 on 4 and 1085141 DF,  p-value: < 2.2e-16
"""

#Le RÂ² est 0.225

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
#y_scale_test (valeur rÃ©elle de la variable de sortie) et prediction_lm_Usualdata_clean_test (valeur estimÃ©e de la variable de sortie)
# en numÃ©rique   
#****************#

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE
#****************#
# TO BE DONE
#****************#
       
# Quelle est la qualitÃ© de la prÃ©diction sur le jeu de test ?

#Le coefficient de dÃ©termination RÂ² est faible, Ã  0.225, donc la qualite de prediction est faible.
#On peut observer le graphique de la variable y prÃ©vue par le modele par rapport Ã  la variable y reelle,
#sur le jeu de test,avec le code suivant:
plot(y_output_test,prediction_lm_Usualdata_clean_test,pch=19,cex=0.8)

#
# QUESTION 6 - REGRESSION LOGISTIQUE
# 
### Q6.1 - Mener une rÃ©gression logisitique de la sortie "fare_amount" (aprÃ¨s binarisation selon la mÃ©diane) 
###        en fonction de l'entrÃ©e (mise Ã  l'Ã©chelle), sur tout le jeu de donnÃ©es


# CrÃ©er la sortie binaire 'fare_binaire' en prenant la valeur mÃ©diane de "fare_amount" comme seuil


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


# Mener la rÃ©gression logistique de "fare_binaire" en fonction des entrÃ©es standardisÃ©es


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


### Q6.2 - Que pouvez-vous dire des rÃ©sultats du modÃ¨le? Quelles variables sont significatives?

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
Signif. codes:  0 â€˜***â€™ 0.001 â€˜**â€™ 0.01 â€˜*â€™ 0.05 â€˜.â€™ 0.1 â€˜ â€™ 1

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

Nous observons que les 4 variables son significatives pour le modÃ¨le normal.
Dans le cas du modÃ¨le big data, la variable dropoff_latitude n'est pas significative
Quand le lieu de dÃ©part est plus Ã  l'est la probabilitÃ© d'avoir une tariffe Ã©lÃ©vÃ© (long/large voyage) sera plus haute.
Quand le lieu d'arrivÃ© est plus Ã  l'est la probabilitÃ© d'avoir une tariffe Ã©lÃ©vÃ© (long/large voyage) sera plus haute.
Le cas contraire s'observe pour les voyages sud-nord ou nord-sud.

"""

### Q6.3 - PrÃ©dire la probabilitÃ© que la course soit plus Ã©levÃ©e que la mÃ©diane
#           en fonction de nouvelles entrÃ©es avec une rÃ©gression linÃ©aire


# Diviser le jeu de donnÃ©es initial en Ã©chantillons d'apprentissage (60% des donnÃ©es), validation (20%) et test (20%)


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


# RÃ©aliser la rÃ©gression logistique sur l'Ã©chantillon d'apprentissage et en testant plusieurs valeurs
# de rÃ©gularisation (hyperpaUsualÃ¨tre de la rÃ©gression logistique) sur l'Ã©chantillon de validation. 


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


# Calculer la prÃ©cision (accuracy) et l'AUC de la prÃ©diction sur le jeu de test.


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


# Quelle est la qualitÃ© de la prÃ©diction sur le jeu de test ?


"""
La qualitÃ© de la prÃ©diction sur le jeu de teste est moyenne.
Avec un accuracy de 62% et un AUC de 60%.
Il existe d'autres variables qui doivent avoir un effet sur le tariff, par exemple: durÃ©e, type de tariff, etc.
"""

#
# QUESTION 7 - RESEAU DE NEURONES (QUESTION BONUS)
# 
### Q7.1 - Mener une régression de la sortie "fare_amount" en fonction de l'entrée (mise à l'échelle), 
###       sur tout le jeu de données, avec un réseau à 2 couches cachées de 10 neurones chacune

# ---------- Utiliser une librairie usuelle
Usualdata.scaled<-as.data.frame(scale(Usualdata_reduit))

vis.nnet<-nnet(fare_amount~.,data=Usualdata.scaled,size=10,decay=2,maxit=400) # maxit opt=400
pred.vistest<-predict(vis.nnet,Usualdata.scaled)

plot(Usualdata.scaled$fare_amount,pred.vistest,main="Neural network predictions vs actual",
     xlab ="Actual",ylab="Predictions")

abline(a=0,b=1, col ="red")

mean((pred.vistest-Usualdata.scaled$fare_amount)^2) # MSE = 0.6443996 faible.

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)





### Q7.2 - Prédire le prix de la course en fonction de nouvelles entrées avec le réseau de neurones entrainé

# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)

# ---------- Utiliser une librairie usuelle
vect <- 1:10000
set.seed(198560)
id4 <- sample(vect,floor(10000*.6))
ech_app <- Usualdata.scaled[id4,]
vect2 <- vect[-id4]
id5 <- sample(vect2,floor(10000*.2))
ech_val <- Usualdata.scaled[id5,]
ech_test <- Usualdata_red_scaled[setdiff(vect2,id5),]

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Réaliser la régression avec réseau de neurones sur l'échantillon d'apprentissage et en testant plusieurs 
# nombre de couches et de neurones par couche sur l'échantillon de validation. 
# ---------- Utiliser une librairie usuelle
# observation du nombre de neurones pour chaque couche caché (2 couchés)
MSE <- numeric(20)
for (k in 1:20) {
  vis.nnet<-nnet(fare_amount~.,data=ech_app,size=k,hidden=2,decay=2,maxit=300) # maxit opt=400
  pred.vistest<-predict(vis.nnet,ech_val)
  MSE[k] <- mean((pred.vistest-ech_val$fare_amount)^2)
}
plot(1:20,MSE,type = "l",col="green",xlab = "Nb de neurones",ylab = "Error",
     main = "Détermination du nombre de neurrone optimale")

s=1:20
size_opt <-s[which(MSE==min(MSE))] # le nombre de neurones optimale est 14

# Détermination du nombre de couche cachée optimale hidden_opt
MSE <- numeric(10)
for (k in 1:10) {
  vis.nnet<-nnet(fare_amount~.,data=ech_app,size=14,hidden=k,decay=2,maxit=300) # maxit opt=400
  pred.vistest<-predict(vis.nnet,ech_val)
  MSE[k] <- mean((pred.vistest-ech_val$fare_amount)^2)
}
plot(1:10,MSE,type = "l",col="green",xlab = "Nb de couches cachées",ylab = "Error",
     main = "Détermination du nombre de couches optimales")

s=1:10
hidden_opt <-s[which(MSE==min(MSE))] # nombre de couches cachées optimale 6

# meilleur modèle après validation
vis.nnet_final<-nnet(fare_amount~.,data=ech_app,size=14,hidden=6,decay=2,maxit=300)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Calculer le RMSE et le R² de la meilleure prédiction sur le jeu de test.
pred.vistest <- predict(vis.nnet_final,ech_test)
RMSE <- mean((pred.vistest-ech_test$fare_amount)^2)


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE

# Quelle est la qualité de la prédiction sur le jeu de test ? Comment se compare-t-elle Ã  la régression linéaire?

plot(ech_test$fare_amount,pred.vistest,main="Prédictions sur l'échantillon test",
     xlab ="Actual",ylab="Predictions")
abline(a=0,b=1, col ="red")

"
La prédiction du prix de taxi sur la jeu de données test est relativement satisfaisante.
La classification de ces prix avant la prédiction pourrait augmente la performance du modèle car
les corrélations entre les variables entrées et la variable de sortie ne sont élevées.
"


