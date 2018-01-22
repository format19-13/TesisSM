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

##Buscar edad en bio y guardarla en el usuario si existe
print "#################################"
print "Ejecutando ml para custom fields"
print "#################################"
main_customFields()

print "################################################################"
print "Ejecutando ml para subscriptionsBOW sobre listas de suscripcion"
print "################################################################"
main_subscriptionBOW()
#comparar resultados/accuracy contra profile pic

print "########################################"
print "Ejecutando ml para featBOW sobre tweets"
print "########################################"
main_featBOW()

print "###########################################"
print "Ejecutando ml para featBigram sobre tweets"
print "###########################################"
main_featBigram()