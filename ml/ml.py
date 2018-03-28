#!/usr/bin/python
# -*- coding: utf8 -*-
from __future__ import division
import os, sys
sys.path.append(os.path.abspath(os.pardir))

from configs.settings import *
from data_access.mongo_utils import MongoDBUtils
from ml_customFields.ml_customFields import main_customFields
from nlp_features.tweetNgrams import main_tweetNgrams
from nlp_features.subscriptionNgrams import main_subscriptionNgrams
from nlp_features.tweetNgramsAndCustomFields import main_tweetNgramsAndCustomFields
import time
import pandas as pd

def getAccuracyFromProfilePic():
	db_access = MongoDBUtils()
	users = db_access.get_users("users")
	cantUsers=0
	cantAciertos=0

	cant1017=0
	cant1824=0
	cant2534=0
	cant3549=0
	cant50xx=0

	for user in users:
		ageRange = user["age"].split('-')
		profilePicAge = user["profile_pic_age"]

		rangeFrom=int(ageRange[0])

		try:
			rangeTo=int(ageRange[1])
		except: 
			rangeTo=100

		if profilePicAge != -1:		
			cantUsers=cantUsers+1
			if (profilePicAge >= rangeFrom and profilePicAge<=rangeTo):
				cantAciertos=cantAciertos+1

			if 10 <= profilePicAge <= 17:
				cant1017 += 1
			elif 18 <= profilePicAge <= 24:
				cant1824 += 1
			elif 25 <= profilePicAge <= 34:
				cant2534 += 1
			elif 35 <= profilePicAge <= 49:
				cant3549 += 1
			elif 50 <= profilePicAge <= 99:
				cant50xx += 1
			else:
				print profilePicAge


	print "Cant users con profile pic age: ", cantUsers
	print "Cant aciertos: ", cantAciertos
	
	print "Cant de users por age group:"
	print '10-17: ', cant1017
	print '18-24: ', cant1824
	print '25-34: ', cant2534
	print '35-49: ', cant3549
	print '50-xx: ', cant50xx

	accuracy=round(cantAciertos/cantUsers,2)
	
	print "Accuracy: ", accuracy
	return accuracy

#print "Calculando accuracy de Profile Pic: " 
#print getAccuracyFromProfilePic()

##Buscar edad en bio y guardarla en el usuario si existe

def runMLAlgorithms(typeOp, balancedFlag):

	print "TIPO ANALISIS: " , typeOp

	print "#################################"
	print "Ejecutando ml para custom fields"
	print "#################################"

	accCustomFields = 0#main_customFields(typeOp,balancedFlag)

	print "########################################"
	print "Ejecutando ml para tweetNgrams sobre tweets"
	print "########################################"
	accTweetNgrams = main_tweetNgrams(typeOp,balancedFlag)

	print "##############################################################"
	print "Ejecutando ml para tweetNgramsAndCustomFields sobre tweets"
	print "##############################################################"
	accTweetNgramsAndCustomFields = 0#main_tweetNgramsAndCustomFields(typeOp,balancedFlag)

	print "################################################################"
	print "Ejecutando ml para subscriptionsBOW sobre listas de suscripcion"
	print "################################################################"
	accSubs = 0#main_subscriptionNgrams(typeOp,balancedFlag)

	print "###########################################"
	print "        ACCURACY DE CADA METODO: "
	print "###########################################"

	print "Custom Fields: ",  accCustomFields
	print '--------------------------------'
	print "Tweets n-grams: ",  accTweetNgrams
	print '--------------------------------'
	print "Tweets n-grams + Custom Fields: ",  accTweetNgramsAndCustomFields
	print '--------------------------------'
	print "Subscription List BOW: ",  accSubs

	df = pd.DataFrame([["Custom Fields", accCustomFields],["Tweets Ngrams", accTweetNgrams], ["Tweets Ngrams+customFields", accTweetNgramsAndCustomFields],["Subscription List Ngrams", accSubs]], columns=['Method','Accuracy'])

	outdir =time.strftime("%d-%m-%Y")+"/"+typeOp
	outname = 'accuracy_'+typeOp+'_'+balancedFlag+'.csv'
	fullname = os.path.join(outdir, outname)    

	import csv
	df.to_csv(fullname,index=False)


runMLAlgorithms('normal', 'unbalanced')
#runMLAlgorithms('pedophilia','unbalanced')
#runMLAlgorithms('normal', 'balanced')
#getAccuracyFromProfilePic()






