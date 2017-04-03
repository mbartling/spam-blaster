#!/usr/bin/env python

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
import glob
import numpy as np
import sys

from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.utils import shuffle
from sklearn import random_projection
import sklearn.decomposition as decomp
import matplotlib.pyplot as plt

import pickle


class Calibrator(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=125, max_depth=3):
        self.base_estimator = CalibratedClassifierCV(XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, \
        colsample_bytree=0.35, subsample=0.9))
    def fit(self, X, y):
        self.base_estimator.fit(X,y)
    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

def main():
    n = 3
    areSpamFiles = glob.glob("data/spam_split/*.txt")
    notSpamFiles = glob.glob("data/not-spam_split/*.txt")
    areSpamLabels = np.ones((len(areSpamFiles),))
    notSpamLabels = np.zeros((len(notSpamFiles),))
    labels = np.concatenate((areSpamLabels, notSpamLabels))
    print labels.shape
    mFiles = []
    mFiles.extend(areSpamFiles)
    mFiles.extend(notSpamFiles)
    docs = []

    for fname in mFiles:
        with open(fname) as fp:
            docs.append(fp.read())
    featureExtractor = HashingVectorizer(decode_error='ignore', analyzer='char', ngram_range=(2,2), norm=None, stop_words='english', non_negative=True)
    #featureExtractor = CountVectorizer(decode_error='ignore', analyzer='char', ngram_range=(2,2))
    randomProjector = random_projection.SparseRandomProjection()
    #with open("featureExtractor.pkl", "rb") as fp:
    #    featureExtractor = pickle.load(fp)
    X = featureExtractor.transform(docs)
    #X = featureExtractor.transform(mFiles)
    #X = randomProjector.fit_transform(X)

    
    #Xcomponents = decomp.TruncatedSVD(2).fit_transform(X)
    #plt.scatter(Xcomponents[0:len(areSpamFiles),0], Xcomponents[0:len(areSpamFiles),1], label="Spam")
    #plt.hold(True)
    #plt.scatter(Xcomponents[len(areSpamFiles):-1,0], Xcomponents[len(areSpamFiles):-1,1], label="Not Spam")
    #plt.legend(["Is Spam", "Is Not Spam"])
    #plt.show()
    #sys.exit(-1)

    

    X, labels, mFiles = shuffle(X, labels, mFiles)
    print X.shape
    #print X
    #sys.exit(0)

    print("Deviation Target = ", 1/np.sqrt(X.shape[0]))

    parameters = {
     #   'finalClassifier__max_depth':[2,5],
     #   'finalClassifier__n_estimators':[100, 150, 200],
        # 'finalClassifier__base_estimator__objective':['binary:logistic']
        # 'finalClassifier__base_estimator': [    XGBClassifier(max_depth=5, n_estimators=200, \
        #   learning_rate=0.0202048,colsample_bytree=0.701, subsample=0.6815, reg_lambda=1.5)\
        #   )\
        # ]
    }
    mPipeline = Pipeline([\
            ("finalClassifier", \
            #CalibratedClassifierCV(\
            #XGBClassifier(max_depth=1, n_estimators=10,
            #    colsample_bytree=0.9, subsample=0.5, reg_lambda=1.1)\
            MultinomialNB()
            #)\
            )])
    print mPipeline
    # In case we actually care to try different parameters
    # Also handles the stratified K-fold for us 
    mGridSearch = GridSearchCV(mPipeline, parameters, iid=False, verbose=1, scoring='roc_auc', cv=n, n_jobs=1)
    mGridSearch.fit(X, list(labels))
    #with open("SpamBlasterModel.pkl", "rb") as fp:
    #    mGridSearch = pickle.load(fp)

    # Could do some stacking here if we wanted to
    print "Mean auc Score, stddev, params"
    for params, mean_score, scores in mGridSearch.grid_scores_:
        print("%0.4f (+/-%0.4f) for %r" % (scores.mean(), scores.std(), params))

    testTxt = []
    with open("notspam.txt") as fp:
        testTxt.append(fp.read())
    test = featureExtractor.transform(testTxt)
    #test = randomProjector.transform(test)
    print "testing notspam.txt", mGridSearch.predict_proba(test)
    
    testTxt = []
    with open("massage.txt") as fp:
        testTxt.append(fp.read())
    test = featureExtractor.transform(testTxt)
    #test = randomProjector.transform(test)
    print "testing massage.txt", mGridSearch.predict_proba(test)
    
    testTxt = []
    with open("isSpam.txt") as fp:
        testTxt.append(fp.read())
    test = featureExtractor.transform(testTxt)
    #test = randomProjector.transform(test)
    print "testing isSpam.txt", mGridSearch.predict_proba(test)

    with open("featureExtractor.pkl", "wb") as fp:
        pickle.dump(featureExtractor, fp)
    
    with open("SpamBlasterModel.pkl", "wb") as fp:
        pickle.dump(mGridSearch, fp)

def classifyBlob(txt):

    with open("featureExtractor.pkl", "rb") as fp:
        featureExtractor = pickle.load(fp)
    
    with open("SpamBlasterModel.pkl", "rb") as fp:
        mGridSearch = pickle.load(fp)
    
    featureExtractor = featureExtractor.set_params(input='content')
    test = featureExtractor.transform([txt])
    print "[[Probability not spam, Probability spam]]"
    print mGridSearch.predict_proba(test)

if __name__ == "__main__":
    main()
    print "Done"
