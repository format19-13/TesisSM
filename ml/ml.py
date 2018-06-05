#!/usr/bin/python
# -*- coding: utf8 -*-
from __future__ import division
import os, sys
sys.path.append(os.path.abspath(os.pardir))

from configs.settings import *
from ml_customFields.ml_customFields import main_customFields
from nlp_features.tweetNgrams import main_tweetNgrams
from nlp_features.subscriptionNgrams import main_subscriptionNgrams
from nlp_features.tweetNgramsAndCustomFields import main_tweetNgramsAndCustomFields
import time
import pandas as pd


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






