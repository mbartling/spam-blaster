#!/usr/bin/env python

from sklearn.feature_extraction.text import HashingVectorizer
from xgboost import XGBClassifier
import glob
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.utils import shuffle


class Calibrator(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=125, max_depth=3):
        self.base_estimator = CalibratedClassifierCV(XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, \
        colsample_bytree=0.35, subsample=0.9))
    def fit(self, X, y):
        self.base_estimator.fit(X,y)
    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

def main():
    n = 5
    areSpamFiles = glob.glob("data/spam_split/*.txt")
    notSpamFiles = glob.glob("data/not-spam_split/*.txt")
    areSpamLabels = np.ones((len(areSpamFiles),))
    notSpamLabels = np.zeros((len(notSpamFiles),))
    labels = np.concatenate((areSpamLabels, notSpamLabels))
    print labels.shape
    mFiles = []
    mFiles.extend(areSpamFiles)
    mFiles.extend(notSpamFiles)

    featureExtractor = HashingVectorizer(mFiles, decode_error='ignore', analyzer='char', ngram_range=(3,3))
    X = featureExtractor.transform(mFiles)

    X, labels, mFiles = shuffle(X, labels, mFiles)

    print("Deviation Target = ", 1/np.sqrt(X.shape[0]))

    parameters = {
        # 'finalClassifier__base_estimator__max_depth':[2,5],
        # 'finalClassifier__base_estimator__objective':['binary:logistic']
        # 'finalClassifier__base_estimator': [    XGBClassifier(max_depth=5, n_estimators=200, \
        #   learning_rate=0.0202048,colsample_bytree=0.701, subsample=0.6815, reg_lambda=1.5)\
        #   )\
        # ]
    }
    mPipeline = Pipeline([\
            ("finalClassifier", \
            #CalibratedClassifierCV(\
            XGBClassifier(max_depth=5, n_estimators=200,
                colsample_bytree=0.6, subsample=0.9, reg_lambda=1.5)\
            #)\
            )])
    print mPipeline
    # In case we actually care to try different parameters
    # Also handles the stratified K-fold for us 
    mGridSearch = GridSearchCV(mPipeline, parameters, iid=False, verbose=1, scoring='roc_auc', cv=n, n_jobs=1)
    mGridSearch.fit(X, list(labels))

    # Could do some stacking here if we wanted to
    print "Mean auc Score, stddev, params"
    for params, mean_score, scores in mGridSearch.grid_scores_:
        print("%0.4f (+/-%0.4f) for %r" % (scores.mean(), scores.std(), params))


if __name__ == "__main__":
    main()
    print "Done"
