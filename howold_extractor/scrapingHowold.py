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
	users = db_access.get_usersWithNoProfilePicAge()
	cont=0
	for user in users : 
		cont = cont+1
		print "-----------------------"

		profilePic=user["profile_image_url_https"].replace("normal", "400x400")
		age= getAgeGenderFromProfilePicture(user['screen_name'],profilePic)

		print user['screen_name'], ' - ' , age 

		if cont%18==0 :
			print "Esperando..."
			time.sleep(60) 

def getAgeGenderFromProfilePicture(screen_name,image):
	db_access = MongoDBUtils()
	users = db_access.get_users('users')
	KEY = '80420a0d0de14f4d9fa2f1c6027afc38'  # Replace with a valid subscription key (keeping the quotes in place).
	CF.Key.set(KEY)
	#KEY: https://azure.microsoft.com/en-us/try/cognitive-services/?apiSlug=face-api&country=Uruguay&allowContact=true
	#TEST ONLINE: https://westcentralus.dev.cognitive.microsoft.com/docs/services/563879b61984550e40cbbe8d/operations/563879b61984550f30395236/console
	#TUTORIAL: https://docs.microsoft.com/en-us/azure/cognitive-services/face/tutorials/faceapiinpythontutorial

	BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/'
	CF.BaseUrl.set(BASE_URL)

	# You can use this example JPG or replace the URL below with your own URL to a JPEG image.
	img_url = image
	resultAge = -1
	resultGender = -1
	
	try:
		faces = CF.face.detect(img_url,False, False,'age,gender')
		#print faces
		#print "age: ", faces[0]['faceAttributes']['age']
		if len(faces) >0:
			resultAge= int(round(faces[0]['faceAttributes']['age'],0))
			resultGender=faces[0]['faceAttributes']['gender']
	except Exception as ex:
		print "User: ",screen_name," - Error while calculating age from profile pic: ", image
		print ex

	if resultAge ==-1:
		streamer=TwitterStreamer(Twython)
		newProfilePic= streamer.getLatestProfilePic(screen_name,image)
		isNew = image != newProfilePic
		print "profile pic updated: " , isNew

		if isNew:
			resultAge= getAgeGenderFromProfilePicture(screen_name,newProfilePic)
			print resultAge

	db_access.set_profilePic_age_gender_user(screen_name,resultAge,resultGender)
	return resultAge