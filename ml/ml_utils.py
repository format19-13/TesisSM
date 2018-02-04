
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

