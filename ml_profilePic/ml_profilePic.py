
import os,sys
import os.path
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.pardir))
from sklearn.model_selection import train_test_split
from configs.settings import *
from data_access.mongo_utils import MongoDBUtils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import itertools
import time
import matplotlib.pyplot as plt

def main_profilePic():
	## *********ARMO EL DATASET DE TRAIN Y EL DE TEST *********
	db_access = MongoDBUtils()
	users_df = db_access.get_profilePicAgeDataset()

	print users_df

	# Split into training and test set
	# 80% of the input for training and 20% for testing

	train_data=users_df.sample(frac=0.8,random_state=200) 
	test_data=users_df.drop(train_data.index)

	# Show the number of observations for the test and training dataframes
	print 'Number of observations in the training data:', len(train_data)
	print 'Number of observations in the test data:',len(test_data)
		
	# Create a list of the feature column's names (everything but the screen_name and age)
	features = users_df.columns[1:(len(users_df.columns)-1)]


	print features
	# convert age ranges into integers
	y = convertToInt(train_data['age'])

	# Create a random forest Classifier.
	clf = RandomForestClassifier(n_jobs=2, random_state=0)


	# Train the Classifier to take the training features and learn how they relate to the age
	clf.fit(train_data[features], y)

	# Apply the Classifier we trained to the test data
	clf.predict(test_data[features])

	# View the predicted probabilities of the first 10 observations
	clf.predict_proba(test_data[features])[0:10]

	# Create actual english names for the ages for each predicted age range
	preds = convertToCategory(clf.predict(test_data[features]))

	outdir =time.strftime("%d-%m-%Y")
	
	if not os.path.exists(outdir):
   		os.mkdir(outdir)

	# View the ACTUAL age for the first five observations
	#print test_data['age'].head()

	#create confusion matrix: anything on the diagonal was classified correctly and the rest incorrectly.
	cnf_matrix =confusion_matrix(test_data['age'].tolist(), preds)
	#print "Confusion Matrix: ", cnf_matrix

	# Plot non-normalized confusion matrix
	fig2 = plt.figure()
	#plot_confusion_matrix(cnf_matrix, classes=db_access.getAgeRanges(),title='Confusion matrix, without normalization for profile pic')
	outname = 'ml_profilePic_confusionMatrixNotNormalized.png'
	fullname = os.path.join(outdir, outname)    
	fig2.savefig(fullname)

	# Plot normalized confusion matrix
	fig3 = plt.figure()
	#plot_confusion_matrix(cnf_matrix, classes=db_access.getAgeRanges(), normalize=True,title='Normalized confusion matrix for profile pic')
	outname = 'ml_profilePic_confusionMatrixNormalized.png'
	fullname = os.path.join(outdir, outname)    
	fig3.savefig(fullname)

	#plt.show()

	# View a list of the features and their importance scores
	print "Importance of Features: ", list(zip(train_data[features], clf.feature_importances_))

	# Copy the results to a pandas dataframe 
	output = pd.DataFrame( data={"id":test_data["screen_name"], "predicted age":preds,"realAge":test_data["age"]})
	#print output

	# Use pandas to write the comma-separated output file
	outname = 'ml_profilePic_result.csv'
	fullname = os.path.join(outdir, outname)    
	output.to_csv(fullname,index=False)

def convertToInt(ageRanges):
	db_access = MongoDBUtils()
	ages = db_access.getAgeRanges()
	result=[]
	for ar in ageRanges:
		result.append(ages.index(ar))
	return result

def convertToCategory(ageRanges):
	db_access = MongoDBUtils()
	ages = db_access.getAgeRanges()
	result=[]
	for ar in ageRanges:
		result.append(ages[ar].encode("utf-8"))
	return result

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix for profile pic")
    else:
        print('Confusion matrix, without normalization for profile pic')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    main_profilePic()

    #https://chrisalbon.com/machine-learning/random_forest_classifier_example_scikit.html
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
