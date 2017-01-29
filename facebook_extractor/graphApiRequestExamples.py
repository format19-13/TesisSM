#from facegraph import Graph
#g = Graph("9f6b035a6a70c37fd01c8027366f1f6a") #  Access token is optional.


from facepy import GraphAPI
import requests

#
#r = requests.get('https://graph.facebook.com/oauth/access_token?#client_id=1795242464078863&client_secret=9f6b035a6a70c37fd01c8027366f1f6a&grant_type=client_credentials')
#access_token = r.text.split('=')[1]
#print access_token

graph = GraphAPI('EAACEdEose0cBAGKccEIslLDXlzgm62iIz4u4Tl6v6UZAxzrC1iuoJfTbXKrcrpfzJZCzfHfOEEZCWxmFsMxtlCOlnSHfUTfYGZAirQZAEJ2en55s1dQ6GdADoeuMXh6aVPOyBJ1fIwq2VoXcUd8VKfR6fjo6mzkM770KQzWXy8QZDZD')
# access key generada en GraphAPI: https://developers.facebook.com/tools/explorer/

#EJEMPLOS
#print graph.get(path = 'me')
#print graph.get(path = 'me/friends')
#print graph.get(path = 'me?fields=gender,languages,timezone')
#print graph.get(path = 'elpais?fields=about ')

#print graph.get(path = 'veronica.tortorella') ##devuelve ERROR porque no se puede buscar por username.

#PRUEBO CON LIBRERIA REQUESTS, pasa lo mismo
#print requests.get('https://graph.facebook.com/v2.2/veronica.tortorella?&access_token=EAACEdEose0cBAGKccEIslLDXlzgm62iIz4u4Tl6v6UZAxzrC1iuoJfTbXKrcrpfzJZCzfHfOEEZCWxmFsMxtlCOlnSHfUTfYGZAirQZAEJ2en55s1dQ6GdADoeuMXh6aVPOyBJ1fIwq2VoXcUd8VKfR6fjo6mzkM770KQzWXy8QZDZD').json()

print graph.search(term='veronica tortorella',type='user')

#PRUEBO CON LIBRERIA REQUESTS
print requests.get('https://graph.facebook.com/v2.2/search?q=coffee&type=place&center=37.76,-122.427&distance=1000&access_token=EAACEdEose0cBAGZBGvUPU4jjSMbTXtKhkhIeyrVGyyzQcXEVoBjoZCnnbzuK0Ug8G52ttXvMOMw8tgODPdayYrC2S0TcE7KtFHOe69cRjvWaAIuwDXpzDD8HZBp79WZAQLi3PKZAwrIYzrv85PrH5kGgaIcWQ7uao6vft4gPf7AZDZD').json()

