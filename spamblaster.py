#!/usr/bin/env python

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_curve

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
    n = 7
    areSpamFiles = []
    notSpamFiles = []
    areSpamFiles.extend(glob.glob("data/spam_split/*.txt"))
    notSpamFiles.extend(glob.glob("data/not-spam_split/*.txt"))
    
    areSpamFiles.extend(glob.glob("sms_data/spam_split/*.txt"))
    #notSpamFiles.extend(glob.glob("sms_data/not-spam_split/*.txt"))
    
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
    #featureExtractor = HashingVectorizer(decode_error='ignore', analyzer='char', ngram_range=(3,3), norm=None, stop_words='english', non_negative=True)
    #X = featureExtractor.transform(docs)

    # With tfidf
    #featureExtractor = TfidfVectorizer(decode_error='ignore', analyzer='char', ngram_range=(3,3), norm=None, stop_words='english')
    #X = featureExtractor.fit_transform(docs)
    
    #Xcomponents = decomp.TruncatedSVD(2).fit_transform(X)
    #plt.scatter(Xcomponents[0:len(areSpamFiles),0], Xcomponents[0:len(areSpamFiles),1], label="Spam")
    #plt.hold(True)
    #plt.scatter(Xcomponents[len(areSpamFiles):-1,0], Xcomponents[len(areSpamFiles):-1,1], label="Not Spam")
    #plt.legend(["Is Spam", "Is Not Spam"])
    #plt.show()
    #sys.exit(-1)

    

    #X, labels, mFiles = shuffle(X, labels, mFiles)
    #print X.shape
    #print X
    #sys.exit(0)

    print("Deviation Target = ", 1/np.sqrt(len(mFiles)))

    parameters = {
            'finalClassifier__alpha':[0.5, 1.0, 1.5]
            #,'tfidf__use_idf': (True, False)
            #,'tfidf__norm': [ 'l2']
     #   'finalClassifier__max_depth':[2,5],
     #   'finalClassifier__n_estimators':[100, 150, 200],
        # 'finalClassifier__base_estimator__objective':['binary:logistic']
        # 'finalClassifier__base_estimator': [    XGBClassifier(max_depth=5, n_estimators=200, \
        #   learning_rate=0.0202048,colsample_bytree=0.701, subsample=0.6815, reg_lambda=1.5)\
        #   )\
        # ]
    }
    mPipeline = Pipeline([\
            ("featureExtractor", HashingVectorizer(decode_error='ignore', analyzer='char', ngram_range=(3,3), norm=None, stop_words='english', non_negative=True)),
            #("tfidf", TfidfTransformer()),
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
    mGridSearch = GridSearchCV(mPipeline, parameters, iid=False, verbose=1, scoring='roc_auc', cv=n, n_jobs=4)
    mGridSearch.fit(docs, list(labels))
    #with open("SpamBlasterModel.pkl", "rb") as fp:
    #    mGridSearch = pickle.load(fp)

    # Could do some stacking here if we wanted to
    print "Mean auc Score, stddev, params"
    for params, mean_score, scores in mGridSearch.grid_scores_:
        print("%0.4f (+/-%0.4f) for %r" % (scores.mean(), scores.std(), params))

    testTxt = []
    with open("notspam.txt") as fp:
        testTxt.append(fp.read())
    print "testing notspam.txt", mGridSearch.predict_proba(testTxt)
    
    testTxt = []
    with open("massage.txt") as fp:
        testTxt.append(fp.read())
    print "testing massage.txt", mGridSearch.predict_proba(testTxt)
    
    testTxt = []
    with open("isSpam.txt") as fp:
        testTxt.append(fp.read())
    print "testing isSpam.txt", mGridSearch.predict_proba(testTxt)

    with open("pipeline.pkl", "wb") as fp:
        pickle.dump(mGridSearch, fp)
    print mGridSearch.get_params()

    fpr, tpr, _ = roc_curve(list(labels), mGridSearch.predict_proba(docs)[:,1])
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % mGridSearch.best_score_)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
def classifyBlob(txt):

    with open("pipeline.pkl", "rb") as fp:
        mGridSearch = pickle.load(fp)
    
    print "[[Probability not spam, Probability spam]]"
    print mGridSearch.predict_proba([txt])

def classifyBlobList(txt):

    with open("pipeline.pkl", "rb") as fp:
        mGridSearch = pickle.load(fp)
    
    print "[[Probability not spam, Probability spam]]"
    res = mGridSearch.predict_proba(txt)
    print res
    return res


if __name__ == "__main__":
    main()
    print "Done"
