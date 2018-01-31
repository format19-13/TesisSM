#!/usr/bin/python
# -*- coding: utf8 -*-
from __future__ import division
import os,sys
import os.path
from pandas import DataFrame
sys.path.append(os.path.abspath(os.pardir))
from configs.settings import *
from configs.data_bases import *
import re
from pymongo.errors import PyMongoError, ConnectionFailure
import pymongo
import logging
import datetime
import emoji

class MongoDBUtils(object):

    def __init__(self):
        # Cliente a MongoDB
        self.mongo_client = pymongo.MongoClient(host=MONGO_DB_HOST, port=MONGO_DB_PORT)
        self.mongo_client.tesisdb.authenticate(MONGO_DB_USER, MONGO_DB_PASSWORD, mechanism='SCRAM-SHA-1')
        self.logger = logging.getLogger(LOGGING_ROOT_NAME + '.data_access')
        self.logger.info('Initializing module.')

    def save_tweet(self, document, source):

        try:

            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]

            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            colT = db[DB_COL_UNLABELED_TWEETS]
            colU = db[DB_COL_UNLABELED_USERS]

            # FLAGS indicando como se configuro el streamer que trajo el tweet(track_terms, follow o bounding box)
           # document["geolocation"] = True if EnumSource.GEOLOCATION in source else False
           # document["follow"] = True if EnumSource.FOLLOW in source else False
            #document["track_terms"] = True if EnumSource.TRACKTERMS in source else False

            colT.insert_one(document)
            if not self.userExistsInDb(document['user']['screen_name'], DB_COL_UNLABELED_USERS) :
                document['user']['exported']=False
                colU.insert_one(document['user'])
        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except PyMongoError as e:
            self.logger.error('Error while trying to save user account', exc_info=True)

    def get_users(self,collection):

        try:

            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[collection]
            return col.find(no_cursor_timeout=True)

        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except PyMongoError as e:
            self.logger.error('Error while trying to save user account', exc_info=True)

    def get_user(self,collection,screen_name):
        try:

            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[collection]
            regx = re.compile(screen_name, re.IGNORECASE)
            
            for user in col.find({"screen_name": regx}):
                return user

        except Exception as e:
            print "error al buscar usuario: ", screen_name, " en la collection: ", collection
            print e

    def save_user(self, document):
            db = self.mongo_client[MONGO_DB_NAME]
            col = db[DB_COL_USERS]
            regx = re.compile(document["screen_name"], re.IGNORECASE)
            if col.find({"screen_name": regx}).count()> 0 :
                print "User: ", document["screen_name"] , " already exists in DB"
            else: 
                col.insert_one(document)
                print "Se salvo usuario: ", document["screen_name"]
                

    def set_age_user(self, screen_name,age,gender):
            db = self.mongo_client[MONGO_DB_NAME]
            col = db[DB_COL_USERS]
            col.update({'screen_name' : screen_name }, {'$set' : {'age' : age }})

    def save_user_tweets(self, screen_name,tweets):
            db = self.mongo_client[MONGO_DB_NAME]
            col = db[DB_COL_USERS]
            col.update({'screen_name' : screen_name }, {'$set' : {'tweets' : tweets }})
            print "Updated tweets for user: ", screen_name
  
    def save_other_network(self, screen_name,network,value):
            db = self.mongo_client[MONGO_DB_NAME]
            col = db[DB_COL_USERS]
            col.update({'screen_name' : screen_name }, {'$set' : {network : value }})

    def set_profilePic_age_gender_user(self, screen_name,age,gender):
            db = self.mongo_client[MONGO_DB_NAME]
            col = db[DB_COL_USERS]
            print screen_name, " - ",  gender
            col.update({'screen_name' : screen_name }, {'$set' : {'profile_pic_age' : age }})
            col.update({'screen_name' : screen_name }, {'$set' : {'profile_pic_gender' : gender }})
    
    def updateProfilePicture(self, screen_name,imageUrl):
            db = self.mongo_client[MONGO_DB_NAME]
            col = db[DB_COL_USERS]
            col.update({'screen_name' : screen_name }, {'$set' : {'profile_image_url_https' : imageUrl }})

    def get_tweetsText(self):

        try:
            df = DataFrame(columns=('screen_name', 'tweets', 'age'))
            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_USERS]
            count=0
            for user in col.find(): #para cada usuario
                tweetText=""
                for tweet in  user['tweets']:
                    tweetText= tweetText +' '+ tweet['full_text']
                #print user['screen_name']
                #print user['age']
                df.loc[count] = [user['screen_name'],tweetText,user['age']]
                count += 1
            return df

        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except PyMongoError as e:
            self.logger.error('Error while trying to save user account', exc_info=True)

    def populate_mentions_hashtags_urls(self):

        try:
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_USERS]
            cont=0

            emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
            r = re.compile('|'.join(re.escape(p) for p in emojis_list))
            re2="[>]?[:;Xx][']?[-=]?[)(PpDdOo$]|<3"

            for user in col.find(): #para cada usuario
                cont = cont +1

                try:
                    user['qtyEmojis']

                    if user['qtyMentions']==0 and user['qtyHashtags']==0 and user['qtyUrls']==0  and user['qtyEmojis']==0:
                        qtyMentions=0
                        qtyHashtags=0
                        qtyUrls=0
                        qtyEmojis=0
                        print cont ,'-',user['screen_name']

                        for tweet in  user['tweets']:
                                
                            qtyMentions=qtyMentions+len(tweet['entities']['user_mentions'])
                            qtyHashtags=qtyHashtags+len(tweet['entities']['hashtags'])
                            qtyUrls=qtyUrls+len(tweet['entities']['urls'])
                            qtyMentions=0
                            txt=tweet['full_text']
                                
                            qtyEmojis= qtyEmojis + len(r.findall(txt))+len(re.findall(re2,txt))
                            qtyTweets = len(user['tweets'])

                        col.update({'screen_name' : user['screen_name'] }, {'$set' : {'qtyMentions' : round(qtyMentions/qtyTweets,2) }})
                        col.update({'screen_name' : user['screen_name'] }, {'$set' : {'qtyHashtags' : round(qtyHashtags/qtyTweets,2) }})
                        col.update({'screen_name' : user['screen_name'] }, {'$set' : {'qtyUrls'     : round(qtyUrls/qtyTweets,2) }})
                        col.update({'screen_name' : user['screen_name'] }, {'$set' : {'qtyEmojis'     : round(qtyEmojis/qtyTweets,2) }})
                except : 
                    print cont
                    ##Si no se calcularon las cantidades aun, las calculo
                    qtyMentions=0
                    qtyHashtags=0
                    qtyUrls=0
                    qtyEmojis=0
                    for tweet in  user['tweets']:
                        
                        qtyMentions=qtyMentions+len(tweet['entities']['user_mentions'])
                        qtyHashtags=qtyHashtags+len(tweet['entities']['hashtags'])
                        qtyUrls=qtyUrls+len(tweet['entities']['urls'])
                        qtyMentions=0
                        txt=tweet['full_text']
                        
                        qtyEmojis= qtyEmojis + len(r.findall(txt))+len(re.findall(re2,txt))
                        qtyTweets = len(user['tweets'])

                    col.update({'screen_name' : user['screen_name'] }, {'$set' : {'qtyMentions' : round(qtyMentions/qtyTweets,2) }})
                    col.update({'screen_name' : user['screen_name'] }, {'$set' : {'qtyHashtags' : round(qtyHashtags/qtyTweets,2) }})
                    col.update({'screen_name' : user['screen_name'] }, {'$set' : {'qtyUrls'     : round(qtyUrls/qtyTweets,2) }})
                    col.update({'screen_name' : user['screen_name'] }, {'$set' : {'qtyEmojis'     : round(qtyEmojis/qtyTweets,2) }})

        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except PyMongoError as e:
            self.logger.error('Error while trying to save user account', exc_info=True)

    def get_tweetsTextForBigrams(self):

        try:
            df = DataFrame(columns=('screen_name', 'tweets', 'age'))
            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_USERS]
            count=0
            for user in col.find(): #para cada usuario
                tweetText=""
                for tweet in  user['tweets']:
                    tweetText= tweetText +'. '+ tweet['full_text']
                #print user['screen_name']
                #print user['age']
                df.loc[count] = [user['screen_name'],tweetText,user['age']]
                count += 1
            return df

        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except PyMongoError as e:
            self.logger.error('Error while trying to save user account', exc_info=True)

    def get_SubscriptionLists(self):
        df = DataFrame(columns=('screen_name', 'subscriptionLists', 'age'))
        # Obtiene una referencia a la instancia de la DB
        db = self.mongo_client[MONGO_DB_NAME]
        # Obtiene el ObjectID Mongo del perfil del data source para el usuario
        col = db[DB_COL_USERS]
        count=0
           
        for user in col.find(): #para cada usuario
            subsLists=""
            try:
                if user['listsSubscriptions'] != -1 and len(user['listsSubscriptions']) >0:
                    for lstSub in user['listsSubscriptions']:
                        subsLists= subsLists +' '+ lstSub['name']
            except Exception as e:
                print user['screen_name']
                print e
            
            df.loc[count] = [user['screen_name'],subsLists,user['age']]
            count += 1
        return df


    def get_tweetsTextFromAgeRange(self, ageRange):

        try:          
            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_USERS]
            tweetText=""
            for user in col.find({"age":ageRange}) :
                for tweet in  user['tweets']:
                    tweetText= tweetText +' '+ tweet['full_text']
            return tweetText        
        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except:
            self.logger.error('Error', exc_info=True)

    def get_SubscriptionListsFromAgeRange(self, ageRange):

        try:          
            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_USERS]
            result=""
            for user in col.find({"age":ageRange}) :
                try:
                    if user['listsSubscriptions'] != -1:
                        for subslist in  user['listsSubscriptions']:
                            
                            if len(subslist)>0:
                                result = result +' '+ (subslist['name'].replace(" ", "_"))
                except: 
                    #print 'usuario: ',user['screen_name'],' no tiene listas'
                    pass
            return result        
        except ConnectionFailure as e:
            print 'Mongo connection error', e
        except Exception as ex:
            print 'Error',ex

    def getAgeRanges(self):

        try:          
            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_USERS]
            ages = col.distinct( "age" ) 
            ages.sort()
            return ages
        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except PyMongoError as e:
            self.logger.error('Error while trying to save user account', exc_info=True)

    def get_customFields(self):

        try:
            df = DataFrame(columns=('screen_name', 'friends_count',  'tweets_count', 'linkedin', 'snapchat', 'instagram','facebook','followers_count','favourites_count','qtyMentions','qtyHashtags','qtyUrls', 'qtyEmojis', 'profile_pic_gender','age'))
            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_USERS]
            count=0
            for user in col.find(): #para cada usuario
                gender=0
                if user['profile_pic_gender'] == 'male' :
                    gender=1
                elif user['profile_pic_gender'] == 'female' :
                    gender=2
                df.loc[count] = [user['screen_name'],user['friends_count'],user['statuses_count'],user['linkedin'],user['snapchat'],user['instagram'],user['facebook'],user['followers_count'],user['favourites_count'], user['qtyMentions'],user['qtyHashtags'],user['qtyUrls'], user['qtyEmojis'], gender,user['age'] ]
                count += 1
            return df

        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except PyMongoError as e:
            self.logger.error('Error while trying to save user account', exc_info=True)


    def get_profilePicAgeDataset(self):

        try:
            df = DataFrame(columns=('screen_name', 'profile_pic_age', 'age'))
            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_USERS]
            count=0
            for user in col.find(): #para cada usuario
                df.loc[count] = [user['screen_name'],user['profile_pic_age'], user['age'] ]
                count += 1
            return df

        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except PyMongoError as e:
            self.logger.error('Error while trying to save user account', exc_info=True)


    def save_listSubscriptions(self, screen_name,lists):
        db = self.mongo_client[MONGO_DB_NAME]
        col = db[DB_COL_USERS]
        col.update({'screen_name' : screen_name }, {'$set' : {'listsSubscriptions' : lists }})
    
    def userExistsInDb(self, screen_name, collection):
        db = self.mongo_client[MONGO_DB_NAME]
        col = db[collection]
        regx = re.compile(screen_name, re.IGNORECASE)
        return col.find({"screen_name": regx}).count()> 0

    def getBioWithAge(self, collection):
        df = DataFrame(columns=('screen_name', 'age'))
        count=0
        db = self.mongo_client[MONGO_DB_NAME]
        col = db[collection]
        screen_names=""
        pathConfig= DIR_PREFIX+'/proyectos/TesisVT/configs/settings.py'

        for user in col.find({'age':{'$exists': False}}): #para cada unlabeled user
            bio = user['description']

            if user['lang'] == 'es' and isinstance(bio, unicode):
                bio = bio.lower()

                if bio.find(u'años')!= -1: 
                    myre = re.compile(ur"(\d+)\s*años",re.UNICODE)

                    match = re.search(myre, bio)
                    if match:
                        screen_name = user["screen_name"]
                        age= int(match.group(1))
                        if age > 9 and age < 100 and screen_name not in df.screen_name.values:
                            if 10 <= age <= 17:
                                ageRange = '10-17'
                            if 18 <= age <= 24:
                                ageRange = '18-24'
                            elif 25 <= age <= 34:
                                ageRange = '25-34'
                            elif 35 <= age <= 49:
                                ageRange = '35-49'
                            elif 50 <= age <= 64:
                                ageRange = '50-64'
                            elif 65 <= age <= 99: 
                                ageRange = '65-xx'
                               
                            print "guardando...",screen_name
                            df.loc[count] = [screen_name,ageRange]
 
                            col.update({'screen_name' : screen_name }, {'$set' : {'age' : age }}) 
                            col.update({'screen_name' : screen_name }, {'$set' : {'ageRange' : ageRange }})                       
        df.to_csv('labeledUsers.csv', index=False)

    def getUrlsFromBio(self, collection):
        count=0
        db = self.mongo_client[MONGO_DB_NAME]
        col = db[collection]
        screen_names=""                
        pathConfig= DIR_PREFIX+'/proyectos/TesisVT/configs/settings.py'

        for user in col.find(): #para cada unlabeled user
            bio = user['description']

            if user['lang'] == 'es' and isinstance(bio, unicode):
                bio = bio.lower()
                screen_name = user["screen_name"]
                
                if (bio.find(u'instagram')!= -1 or bio.find(u'ig:')!= -1 or bio.find(u'insta')!= -1) and (not user["instagram"]) : 
                    col.update({'screen_name' : screen_name }, {'$set' : {'instagram' : True }}) 
                    print "Usuario: ", screen_name, " tiene instagram en la bio"
                if (bio.find(u'snap')!= -1 or bio.find(u'snapchat:')!= -1) and (not user["snapchat"]): 
                    col.update({'screen_name' : screen_name }, {'$set' : {'snapchat' : True }}) 
                    print "Usuario: ", screen_name, " tiene snapchat en la bio"
                if bio.find(u'linkedin')!= -1 and (not user["linkedin"]): 
                    col.update({'screen_name' : screen_name }, {'$set' : {'linkedin' : True }}) 
                    print "Usuario: ", screen_name, " tiene linkedin en la bio"

    def getEdad(self,screen_name,collection):
        db = self.mongo_client[MONGO_DB_NAME]
        col = db[collection]
        age = -1

        regx = re.compile(screen_name, re.IGNORECASE)

        for user in col.find({"screen_name": regx}):
            try:
                age=user['ageRange']
            except:
                pass
        #if age == -1 : 
            #print "Usuario: ", screen_name, ", edad: La edad no pudo ser obtenida de unlabeled_users"

        return age

    def populateInvalidUserAges(self):
        db = self.mongo_client[MONGO_DB_NAME]
        col = db["users"]
        age = -1

        for user in col.find({"age": -1}):
            regx = re.compile(user["screen_name"], re.IGNORECASE)
            try:
                age = self.getEdad(user['screen_name'],"unlabeled_users")
                print user["screen_name"] , ': edad: ', age
                col.update({'screen_name' : user["screen_name"] }, {'$set' : {'age' : age }})
            except:
                pass
            
        for user in col.find({'age':{'$exists': False}}):
            regx = re.compile(user["screen_name"], re.IGNORECASE)
            try:
                age = self.getEdad(user['screen_name'],"unlabeled_users")
                print user["screen_name"] , ': edad: ', age
                col.update({'screen_name' : user["screen_name"] }, {'$set' : {'age' : age }})
            except:
                pass


    def populateUsersOfTweetsStreamed(self):
        db = self.mongo_client[MONGO_DB_NAME]
        colT = db['unlabeled_tweets']
        colU=    db['unlabeled_users']

        for tweet in colT.find(): #para cada tweet
            if not self.userExistsInDb(tweet['user']['screen_name'],'unlabeled_users'):            
                colU.insert_one(tweet['user'])
                print "insertando en DB: ", tweet['user']['screen_name']

    def hasValidProfilePicAge(self,screen_name):
        db = self.mongo_client[MONGO_DB_NAME]
        col = db['users']
        regx = re.compile(screen_name, re.IGNORECASE)
        user = col.find({"screen_name": regx})[0]
        
        try:
            user['profile_pic_age']
            return True #user["profile_pic_age"] != -1
        except:
            return False

    def export_tweetsLabeled(self):
        # Obtiene una referencia a la instancia de la DB
        db = self.mongo_client[MONGO_DB_NAME]
        # Obtiene el ObjectID Mongo del perfil del data source para el usuario
        col = db[DB_COL_USERS]

        colUnlabeled = db["unlabeled_users"]

        df = DataFrame(columns=('screen_name', 'tweet text','age', 'ageRange','profile_pic_age'))
        #print 'screen_name',',', 'tweet text',',','age',',', 'ageRange',',','profile_pic_age'
        count=0
        for user in col.find(): #para cada usuario
            ageReal=-1
            try:
                regx = re.compile(user['screen_name'], re.IGNORECASE)
                userLabeled=colUnlabeled.find({"screen_name": regx})

                for userL in userLabeled:
                    ageReal=userL['age']

            except Exception as a:
                #print "Error with user: ", user['screen_name'], " while getting real age"
                #print a
                pass

            for tweet in  user['tweets']:
                try:
                    df.loc[count] = [user['screen_name'],tweet['full_text'],ageReal, user['age'],user['profile_pic_age'] ]
                    #print user['screen_name'],",",tweet['text'],",",ageReal,",", user['age'],",",user['profile_pic_age']
                    count += 1
                except Exception as a:
                    #print "Error with user: ", user['screen_name'] 
                    #print a
                    pass
        print "Generando el archivo tweetsLabeled.csv.."
        df.to_csv('tweetsLabeled.csv', index=False, encoding='utf-8')

    def get_unlabeled_users_with_age(self):

        # Obtiene una referencia a la instancia de la DB
        db = self.mongo_client[MONGO_DB_NAME]
        # Obtiene el ObjectID Mongo del perfil del data source para el usuario
        col = db["unlabeled_users"]
        return col.find({'age':{'$exists': True},'exported':False}, no_cursor_timeout=True)

    def markUnlabeledAsLabeled(self,userUnlabeled):
        print "Marco como labeled: ",userUnlabeled["screen_name"]
        print "ANTES: ", userUnlabeled["exported"]
         # Obtiene una referencia a la instancia de la DB
        db = self.mongo_client[MONGO_DB_NAME]
        # Obtiene el ObjectID Mongo del perfil del data source para el usuario
        col = db["unlabeled_users"]
        col.update({'screen_name' : userUnlabeled["screen_name"] }, {'$set' : {'exported' : True }})
        print "AHORA: ",userUnlabeled["exported"]


    def aux(self):
        db = self.mongo_client[MONGO_DB_NAME]
        col = db["unlabeled_users"]
        i=0
        for user in self.get_unlabeled_users_with_age():
            i=i+1
            print i
            if self.userExistsInDb(user['screen_name'], "users") :
                print user['screen_name'], "EXISTE!!!"
                col.update({'screen_name' : user["screen_name"] }, {'$set' : {'exported' : True }})
            else:
                col.update({'screen_name' : user["screen_name"] }, {'$set' : {'exported' : False }})
                print user['screen_name'], "NO EXISTE"

    def get_usersWithNoProfilePicAge(self):
        # Obtiene una referencia a la instancia de la DB
        db = self.mongo_client[MONGO_DB_NAME]
        # Obtiene el ObjectID Mongo del perfil del data source para el usuario
        col = db['users']
        return col.find({'profile_pic_age':{'$exists': False}}, no_cursor_timeout=True)

    def get_usersWithNoSubscriptionLists(self):
        # Obtiene una referencia a la instancia de la DB
        db = self.mongo_client[MONGO_DB_NAME]
        # Obtiene el ObjectID Mongo del perfil del data source para el usuario
        col = db['users']
        return col.find({'listsSubscriptions':{'$exists': False}}, no_cursor_timeout=True)

