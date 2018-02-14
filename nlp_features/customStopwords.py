# coding=utf-8
# This Python file uses the following encoding: utf-8
import os,sys
reload(sys)
import pandas as pd
sys.path.append(os.path.abspath(os.pardir))
sys.setdefaultencoding('utf8')
sys.path.append(os.path.abspath(os.pardir))
from configs.settings import *
from data_access.mongo_utils import MongoDBUtils
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

def getCustomStopwords():
	
	#generateCustomStopWordsForallAgeRanges()
	stopset = stopwords.words('spanish')

	with open(NLP_PATH+'/customStopwords.csv', 'rb') as csvfile:
		customStopwords = csv.reader(csvfile, delimiter=',')
		for row in customStopwords:
			stopset+=row

	listaResultFinal=[]
	for st in stopset:
		if not isinstance(st, unicode):
			listaResultFinal.append(unicode(st))
		else:
			listaResultFinal.append(st)
	return sorted(listaResultFinal)

def getSpanishStopwords():
	
	stopset = stopwords.words('spanish')
	return sorted(stopset)

def obtainMostFrequentWordsInAgeRange(ar):
	print "Obtaining most frequent words for age range: ", ar
	print '-----------------------------------------------------'
	df_tweets=pd.read_csv(DATASET_PATH+"/tweets_"+ar+".csv", sep=",",dtype=str)

	stopset =(stopwords.words('spanish')) 

	transformer_tfidf = TfidfVectorizer(smooth_idf=False,lowercase=False,stop_words=stopset,max_features=5000)
	tfidf = transformer_tfidf.fit_transform(df_tweets.tweets)

	idf = transformer_tfidf.idf_
	
	valuesTfIdf = sorted(zip(idf,transformer_tfidf.get_feature_names()), key=lambda x: x[0])[:200]
	frequents = zip(*valuesTfIdf)[1]
	print frequents
	return sorted(frequents)

def generateCustomStopWordsForallAgeRanges():
	print "Obtaining most frequent words foreach age range... "
	
	words1017=obtainMostFrequentWordsInAgeRange('10-17')
	words1824=obtainMostFrequentWordsInAgeRange('18-24')
	words2534=obtainMostFrequentWordsInAgeRange('25-34')
	words3549=obtainMostFrequentWordsInAgeRange('35-49')
	words50xx=obtainMostFrequentWordsInAgeRange('50-xx')

	dictionary=set(words1017+words1824+words2534+words3549+words50xx)

	result =[]
	
	for word in dictionary:
		cont=0
		
		if unicode(word) in words1017:
			cont+=1
		if unicode(word) in words1824:
			cont+=1	
		if unicode(word) in words2534:
			cont+=1	
		if unicode(word) in words3549:
			cont+=1	
		if unicode(word) in words50xx:
			cont+=1	

		if cont>=3:
			result.append(word)

	with open(NLP_PATH+'/customStopwords.csv', 'wb') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		wr.writerow(sorted(result))

	print '-----------------------------------------------------'
	print "NUEVAS STOPWORDS: "
	print sorted(result)
	return sorted(result)
