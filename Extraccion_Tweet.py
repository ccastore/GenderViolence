import tweepy
import json
import os


class TweesListerner(tweepy.StreamListener):
	def on_connect(self):
		print("Conected")

	def on_status(self,status):
		print(status.text)
		if len(status.text)>40:
			#the archive name to writte the information
			data=open("/home/carlos/Escritorio/ArticuloVI/Data.txt",'a')
			data.write(str(status.text))
			data.write(os.linesep)
			data.close()

	def on_error (self, status_code):
		print("Error", status_code)


#Validation KEY of twitter
consumer_key = "" #Counsumer Key 
consumer_secret = "" #Consumer Key Secret
access_token ="" #Access Token
access_token_secret= "" #Access Token Secret

auth = tweepy.OAuthHandler (consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


stream= TweesListerner()
streamingApi = tweepy.Stream(auth=api.auth, listener=stream)
streamingApi.filter(
	track=["Mujeres", "Violencia", "Genero"], #filter by key words
	locations=[-118.59919006,14.38862422,-86.49327807,32.71865523], #filter by locations
	languages=["es"], #filter by key lenguaje
)