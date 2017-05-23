
import os,sys
import os.path
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.pardir))
from sklearn.model_selection import train_test_split
from configs.settings import *
from data_access.mongo_utils import MongoDBUtils


db_access = MongoDBUtils()
users_df = db_access.get_tweetsText()

train_data=users_df.sample(frac=0.9,random_state=200)
test_data=users_df.drop(train_data.index)

print len(train_data)
print len(test_data)