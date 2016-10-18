import requests
import facebook

app_id= '1795242464078863'
app_secret= '9f6b035a6a70c37fd01c8027366f1f6a'
payload = {'grant_type': 'client_credentials', 'client_id': app_id, 'client_secret': app_secret}
file = requests.post('https://graph.facebook.com/oauth/access_token?', params = payload)
result = file.text.split("=")[1]
print result

token = result
graph = facebook.GraphAPI(token)
data = graph.request('/search?q=veronicatortorella&type=user')
print data

