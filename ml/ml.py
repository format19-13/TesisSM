#!/usr/bin/python
# -*- coding: utf8 -*-
import os, sys
sys.path.append(os.path.abspath(os.pardir))

from configs.settings import *
from data_access.mongo_utils import MongoDBUtils
from ml_customFields.ml_customFields import main_customFields
from ml_profilePic.ml_profilePic import main_profilePic
from nlp_features.featBOW import main_featBOW
from nlp_features.featBigram import main_featBigram

##Buscar edad en bio y guardarla en el usuario si existe
print "#################################"
print "Ejecutando ml para custom fields"
print "#################################"
main_customFields()

print "#################################"
print "Ejecutando ml para profile pic"
print "#################################"
main_profilePic()

print "########################################"
print "Ejecutando ml para featBOW sobre tweets"
print "########################################"
main_featBOW()

print "###########################################"
print "Ejecutando ml para featBigram sobre tweets"
print "###########################################"
main_featBigram()

##Mover usuarios etiquetados en paso anterior a la collection "users"
#print "Ejecutando extractUsers.py"
#processor = TwitterStreamer(source=SOURCE)
#processor.run()

##guardar la edad en base a la profile pic de la collection "users"
#print "Ejecutando analyzeProfilePicture.py"
#analyzeProfilePicture()

##guardar las subscriptions del user en collection "users"
#print "Ejecutando extractListsSubscriptions.py"
#lists = TwitterStreamerSubscriptions(source=SOURCE)
#lists.run()
