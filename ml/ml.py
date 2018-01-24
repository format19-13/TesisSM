#!/usr/bin/python
# -*- coding: utf8 -*-
import os, sys
sys.path.append(os.path.abspath(os.pardir))

from configs.settings import *
from data_access.mongo_utils import MongoDBUtils
from ml_customFields.ml_customFields import main_customFields
from nlp_features.featBOW import main_featBOW
from nlp_features.subscriptionBOW import main_subscriptionBOW
from nlp_features.featBigram import main_featBigram
import time
import pandas as pd

##Buscar edad en bio y guardarla en el usuario si existe
print "#################################"
print "Ejecutando ml para custom fields"
print "#################################"
accCustomFields = main_customFields()

print "################################################################"
print "Ejecutando ml para subscriptionsBOW sobre listas de suscripcion"
print "################################################################"
accSubs = main_subscriptionBOW()

#comparar resultados/accuracy contra profile pic

print "########################################"
print "Ejecutando ml para featBOW sobre tweets"
print "########################################"
accFeatBOW = main_featBOW()

print "###########################################"
print "Ejecutando ml para featBigram sobre tweets"
print "###########################################"
accFeatBigram = main_featBigram()

print "###########################################"
print "        ACCURACY DE CADA METODO: "
print "###########################################"

print "Custom Fields: ",  accCustomFields
print "Subscription List BOW: ",  accSubs
print "Tweets BOW: ",  accFeatBOW
print "Tweets Bigram: ",  accFeatBigram

df = pd.DataFrame([["Custom Fields", accCustomFields], ["Subscription List BOW", accSubs],["Tweets BOW", accFeatBOW], ["Tweets Bigram", accFeatBigram]], columns=['Method','Accuracy'])

outdir =time.strftime("%d-%m-%Y")
outname = 'accuracy.csv'
fullname = os.path.join(outdir, outname)    

import csv
df.to_csv(fullname,index=False)
