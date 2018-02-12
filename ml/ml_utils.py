
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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB


def convertToInt(ageRanges,typeOp):
    db_access = MongoDBUtils()
    ages=[]
    result=[]
    
    if typeOp =='normal':
        ages = db_access.getAgeRanges()
    else:
        ages=['10-17','18-24','25-xx']

    for ar in ageRanges:
        result.append(ages.index(ar))

    return result

def convertToCategory(ageRanges,typeOp):
    db_access = MongoDBUtils()
    if typeOp =='normal':
        ages = db_access.getAgeRanges()
    else:
        ages=['10-17','18-24','25-xx']
	
    result=[]
    for ar in ageRanges:
        result.append(ages[ar].encode("utf-8"))
    return result

def plot_confusion_matrix(cm, classes,normalize,title,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print title

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

def createConfusionMatrix(y_true,y_pred,classes,className,mlAlgorithm,outdir):
    cnf_matrix =confusion_matrix(y_true, y_pred)

    # Plot non-normalized confusion matrix
    fig1 = plt.figure()
    plot_confusion_matrix(cnf_matrix, classes,normalize=False,
                        title='Confusion matrix, without normalization for '+className+ '-'+ mlAlgorithm)
    
    outname = 'ml_'+className+'_'+ mlAlgorithm+'_confusionMatrixNotNormalized.png'
    fullname = os.path.join(outdir, outname)    
    fig1.savefig(fullname)

    # Plot normalized confusion matrix
    fig2 = plt.figure()
    plot_confusion_matrix(cnf_matrix, classes, normalize=True,
                    title='Normalized confusion matrix for '+className+ '-'+ mlAlgorithm)
    
    outname = 'ml_'+className+'_'+ mlAlgorithm+'_confusionMatrixNormalized.png'
    fullname = os.path.join(outdir, outname)
    fig2.savefig(fullname)

#####################################
##      HYPERPARAMETER TUNING
#####################################

def SVM_param_selection(X, y): 
    print "Tuning parameters for SVM..."
    Cs = [8, 10, 12, 14]
    gammas = [ 0.01, 0.1 , 1]
    kernel=['rbf','linear']
    param_grid = {'C': Cs, 'gamma' : gammas,'kernel':kernel}
    grid_search = GridSearchCV(SVC(), param_grid, cv=10)
    grid_search.fit(X, y)
    grid_search.best_params_
    print "Finished Tuning parameters for SVM..."
    return grid_search.best_params_

def RandomForest_param_selection(X, y): 
    print "Tuning parameters for RandomForest..."

    param_grid = { 
           "n_estimators" : [120, 140,160, 180],
           "max_depth" : [20,25, 30, 35 ],
           "min_samples_leaf" : [1, 2, 3, 4]}
 
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv= 10)
    grid_search.fit(X, y)
    grid_search.best_params_
    print "Finished Tuning parameters for RandomForest..."
    return grid_search.best_params_

def SGD_param_selection(X,y): 
    print "Tuning parameters for SGD..."
    param_grid = {
        'n_iter': [40, 50, 60],
        'loss': ('log', 'hinge'),
        'penalty': ['l1', 'l2', 'elasticnet'],
        'alpha': [0.001, 0.0001, 0.00001]
    }
    grid_search = GridSearchCV(estimator=SGDClassifier(), param_grid=param_grid, cv= 10)
    grid_search.fit(X, y)
    grid_search.best_params_
    print "Finished Tuning parameters for SGD..."
    return grid_search.best_params_
