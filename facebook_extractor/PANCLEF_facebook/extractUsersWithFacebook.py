import os, sys
sys.path.append(os.path.abspath(os.pardir))
import pymongo
from pymongo import MongoClient
from data_access.mongo_utils import MongoDBUtils
import imp

class NLPExtractor():

    def run(self):
        db_access = MongoDBUtils()
        users= db_access.get_users()
 
        for obj in users:
            try:
                urls= obj["entities"]["url"]["urls"]
                for url in urls:
                    if "facebook" in url["expanded_url"]:
                        print url["expanded_url"]
                        foo = imp.load_source('getEdad', '/home/vero/proyectos/TesisVT/facebook_extractor/scrapingFacebook.py')
                        print foo("mcopes")
            except:
                pass

def main():
    print 'Process start...'
    processor = NLPExtractor()
    processor.run()
    print 'Exiting now.'

if __name__ == "__main__":
    main()
