import cognitive_face as CF

def getAgeFromProfilePicture(image):
	KEY = '568aebd6112041fb8055d8e583f78f94'  # Replace with a valid subscription key (keeping the quotes in place).
	CF.Key.set(KEY)
	#KEY: https://azure.microsoft.com/en-us/try/cognitive-services/?apiSlug=face-api&country=Uruguay&allowContact=true
	#TEST ONLINE: https://westcentralus.dev.cognitive.microsoft.com/docs/services/563879b61984550e40cbbe8d/operations/563879b61984550f30395236/console
	#TUTORIAL: https://docs.microsoft.com/en-us/azure/cognitive-services/face/tutorials/faceapiinpythontutorial

	BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/'
	CF.BaseUrl.set(BASE_URL)

	# You can use this example JPG or replace the URL below with your own URL to a JPEG image.
	img_url = image
	faces = CF.face.detect(img_url,False, False,'age')
	print "age: ", faces[0]['faceAttributes']['age']
	return faces[0]['faceAttributes']['age']

getAgeFromProfilePicture('https://pbs.twimg.com/profile_images/942493717948239874/zR-9ikBm_400x400.jpg')
