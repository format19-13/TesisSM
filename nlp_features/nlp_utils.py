
# coding=utf-8
# This Python file uses the following encoding: utf-8
import os,sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append(os.path.abspath(os.pardir))

from nltk.corpus import stopwords

def generateCustomStopwords():

	stopset = set(stopwords.words('spanish'))

	'''
	##add domain stopwords
	stopset.add((u'hoy'))
	stopset.add((u'más'))
	stopset.add((u'si'))
	stopset.add((u'aquí'))
	stopset.add((u'ahora'))
	stopset.add((u'está'))
	stopset.add((u'ser'))
	stopset.add((u'bien'))
	stopset.add((u'gracias'))
	stopset.add((u'qué'))
	stopset.add((u'día'))
	stopset.add((u'días'))
	stopset.add((u'año'))
	stopset.add((u'años'))
	stopset.add((u'mejor'))
	stopset.add((u'puede'))
	stopset.add((u'hacer'))
	stopset.add((u'así'))
	stopset.add((u'hace'))
	stopset.add((u'ver'))
	stopset.add((u'cómo'))
	stopset.add((u'va'))
	stopset.add((u'españa'))
	stopset.add((u'vía'))
	stopset.add((u'gran'))
	stopset.add((u'nuevo'))*/
	'''
	return stopset
