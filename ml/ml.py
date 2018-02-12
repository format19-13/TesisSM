#!/usr/bin/python
# -*- coding: utf8 -*-
from __future__ import division
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

def getAccuracyFromProfilePic():
	db_access = MongoDBUtils()
	users = db_access.get_users("users")
	cantUsers=0
	cantAciertos=0

	for user in users:
		cantUsers=cantUsers+1
		ageRange = user["age"].split('-')
		profilePicAge = user["profile_pic_age"]

		rangeFrom=int(ageRange[0])

		try:
			rangeTo=int(ageRange[1])
		except: 
			rangeTo=100

		if profilePicAge != -1:		
			if (profilePicAge >= rangeFrom and profilePicAge<=rangeTo):
				cantAciertos=cantAciertos+1

	accuracy=round(cantAciertos/cantUsers,2)
	return accuracy

#print "Calculando accuracy de Profile Pic: " 
#print getAccuracyFromProfilePic()

##Buscar edad en bio y guardarla en el usuario si existe

def runMLAlgorithms(typeOp):

	print "TIPO ANALISIS: " , typeOp

	print "#################################"
	print "Ejecutando ml para custom fields"
	print "#################################"

	accCustomFields = 0#main_customFields(typeOp,'unbalanced')
	accCustomFieldsBalanced = 0#main_customFields(typeOp,'balanced')

	print "################################################################"
	print "Ejecutando ml para subscriptionsBOW sobre listas de suscripcion"
	print "################################################################"
	accSubs = main_subscriptionBOW(typeOp,'unbalanced')
	accSubsBalanced = 0#main_subscriptionBOW(typeOp,'balanced')
	#comparar resultados/accuracy contra profile pic

	print "########################################"
	print "Ejecutando ml para featBOW sobre tweets"
	print "########################################"
	accFeatBOW = 0#main_featBOW(typeOp,'unbalanced')
	accFeatBOWBalanced = 0#main_featBOW(typeOp,'balanced')

	print "###########################################"
	print "Ejecutando ml para featBigram sobre tweets"
	print "###########################################"
	accFeatBigram = 0#main_featBigram(typeOp,'unbalanced')
	accFeatBigramBalanced = 0#main_featBigram(typeOp,'balanced')

	print "###########################################"
	print "        ACCURACY DE CADA METODO: "
	print "###########################################"

	print "Custom Fields: ",  accCustomFields
	print '--------------------------------'
	print "Subscription List BOW: ",  accSubs
	print '--------------------------------'
	print "Tweets BOW: ",  accFeatBOW
	print '--------------------------------'
	print "Tweets Bigram: ",  accFeatBigram

	df = pd.DataFrame([["Custom Fields", accCustomFields], ["Custom Fields Balanced", accCustomFieldsBalanced], ["Subscription List BOW", accSubs],["Subscription List BOW Balanced", accSubsBalanced],["Tweets BOW", accFeatBOW],["Tweets BOW Balanced", accFeatBOWBalanced], ["Tweets Bigram", accFeatBigram],["Tweets Bigram Balanced", accFeatBigramBalanced]], columns=['Method','Accuracy'])

	outdir =time.strftime("%d-%m-%Y")+"/"+typeOp
	outname = 'accuracy_'+typeOp+'.csv'
	fullname = os.path.join(outdir, outname)    

	import csv
	df.to_csv(fullname,index=False)

runMLAlgorithms('normal')
#runMLAlgorithms('pedophilia')







