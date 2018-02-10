#!/usr/bin/python
# -*- coding: utf8 -*-
import os, sys
sys.path.append(os.path.abspath(os.pardir))

from configs.settings import *
from data_access.mongo_utils import MongoDBUtils

##Mover usuarios etiquetados en paso anterior a la collection "users"
#print "Ejecutando extractUsers.py"
db_access = MongoDBUtils()
#db_access.export_tweetsText_toCSV('normal')
#db_access.export_tweetsText_toCSV('pedophilia')

#db_access.export_subscriptionLists_toCSV('normal')
#db_access.export_subscriptionLists_toCSV('pedophilia')

#db_access.export_customFields('normal')
#db_access.export_customFields('pedophilia')

db_access.export_tweetsTextFromAgeRange('10-17')
db_access.export_tweetsTextFromAgeRange('18-24')
db_access.export_tweetsTextFromAgeRange('25-34')
db_access.export_tweetsTextFromAgeRange('35-49')
db_access.export_tweetsTextFromAgeRange('50-64')
db_access.export_tweetsTextFromAgeRange('65-xx')
