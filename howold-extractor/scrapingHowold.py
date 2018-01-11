#!/usr/bin/python
# -*- coding: utf8 -*-
import os, sys
sys.path.append(os.path.abspath(os.pardir))
import time
import cognitive_face as CF
from data_access.mongo_utils import MongoDBUtils
from twitter_timeline_extractor.extractUsers import TwitterStreamer
from twython import Twython

def analyzeProfilePicture():
	db_access = MongoDBUtils()
	users = db_access.get_users('users')
	cont=0
	for user in users : 
		if not db_access.hasValidProfilePicAge(user['screen_name']):
			cont = cont+1
			print "-----------------------"

			profilePic=user["profile_image_url_https"].replace("normal", "400x400")
			age= getAgeFromProfilePicture(user['screen_name'],profilePic)

			print user['screen_name'], ' - ' , age 

			if cont%20==0 :
				print "Esperando..."
				time.sleep(60) 

			db_access.set_profilePic_age_user(user['screen_name'],age)

def getAgeFromProfilePicture(screen_name,image):
	KEY = '568aebd6112041fb8055d8e583f78f94'  # Replace with a valid subscription key (keeping the quotes in place).
	CF.Key.set(KEY)
	#KEY: https://azure.microsoft.com/en-us/try/cognitive-services/?apiSlug=face-api&country=Uruguay&allowContact=true
	#TEST ONLINE: https://westcentralus.dev.cognitive.microsoft.com/docs/services/563879b61984550e40cbbe8d/operations/563879b61984550f30395236/console
	#TUTORIAL: https://docs.microsoft.com/en-us/azure/cognitive-services/face/tutorials/faceapiinpythontutorial

	BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/'
	CF.BaseUrl.set(BASE_URL)

	# You can use this example JPG or replace the URL below with your own URL to a JPEG image.
	img_url = image
	result = -1
	try:
		faces = CF.face.detect(img_url,False, False,'age')
		#print faces
		#print "age: ", faces[0]['faceAttributes']['age']
		if len(faces) >0:
			result= int(round(faces[0]['faceAttributes']['age'],0))
	except Exception as ex:
		print "User: ",screen_name," - Error while calculating age from profile pic: ", image
		print ex

	if result ==-1:
		streamer=TwitterStreamer(Twython)
		newProfilePic= streamer.getLatestProfilePic(screen_name)
		isNew = image != newProfilePic
		print "profile pic updated: " , isNew

		if isNew:
			result= getAgeFromProfilePicture(screen_name,newProfilePic)
			print result
	return result

analyzeProfilePicture()