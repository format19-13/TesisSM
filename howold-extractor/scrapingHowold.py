import cognitive_face as CF

KEY = '568aebd6112041fb8055d8e583f78f94'  # Replace with a valid subscription key (keeping the quotes in place).
CF.Key.set(KEY)
# If you need to, you can change your base API url with:
#CF.BaseUrl.set('https://westcentralus.api.cognitive.microsoft.com/face/v1.0/')

BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/'
CF.BaseUrl.set(BASE_URL)

# You can use this example JPG or replace the URL below with your own URL to a JPEG image.
img_url = 'https://pbs.twimg.com/profile_images/942493717948239874/zR-9ikBm_400x400.jpg'
faces = CF.face.detect(img_url,False, False,'age')
print(faces)