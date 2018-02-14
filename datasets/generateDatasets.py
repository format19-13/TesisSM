#!/usr/bin/python
# -*- coding: utf8 -*-
import os, sys
sys.path.append(os.path.abspath(os.pardir))

from configs.settings import *
from data_access.mongo_utils import MongoDBUtils
from nlp_features.customStopwords import generateCustomStopWordsForallAgeRanges

db_access = MongoDBUtils()

db_access.export_tweetsText_toCSV('normal','')
db_access.export_tweetsText_toCSV('normal','faceAPI')

db_access.export_tweetsText_toCSV('pedophilia','')
db_access.export_tweetsText_toCSV('pedophilia','faceAPI')

db_access.export_customFields('normal','')
db_access.export_customFields('normal','faceAPI')

db_access.export_customFields('pedophilia','')
db_access.export_customFields('pedophilia','faceAPI')

db_access.export_subscriptionLists_toCSV('normal','')
db_access.export_subscriptionLists_toCSV('normal','faceAPI')

db_access.export_subscriptionLists_toCSV('pedophilia','')
db_access.export_subscriptionLists_toCSV('pedophilia','faceAPI')

db_access.export_tweetsText_toCSV_balanced()
db_access.export_customFields_balanced()
db_access.export_subscriptionLists_toCSV_balanced()

ages=db_access.getAgeRanges()

for age in ages:
	db_access.export_tweetsTextFromAgeRange(age)

for age in ages:
	db_access.export_subscriptionListsFromAgeRange(age)

generateCustomStopWordsForallAgeRanges()