# Spam Blaster
 Author: Michael Bartling

# Overview

Spam Blaster is a simple attempt to thwart spam using basic data mining techniques. The idea is to train some bag of words classifier to estimate the probability a sample was taken from some language distribution. The feature vectors are sentences in a document and the document is classified as spam/not-spam by ensembling the individual sentence predictions.
