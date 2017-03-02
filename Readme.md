# Spam Blaster
 Author: Michael Bartling

# Overview

Spam Blaster is a simple attempt to thwart spam using basic data mining techniques. The idea is to train some bag of words classifier to estimate the probability a sample was taken from some language distribution. The feature vectors are sentences in a document and the document is classified as spam/not-spam by ensembling the individual sentence predictions.

# Notes on Spam

Currently, most of the spam is taken directly from the mbed forums. However, this spam is not very diverse and often appears in the form WORD1 WORD1 WORD1 RANDOMWORD WORD1 WORD1 WORD1. Although this makes it easy to detect using signature based approaches, signatures suffer from over specialization and hand crafting. Therefore, I extend the spam feature set to include samples taken from http://untroubled.org/spam/

SpamBlaster operates on sentence level feature vectors where each feature is a vectorized character trigram using the hashing trick. This works because we expect to see common character sequences in real user posts, such as "pin", "read", "write", "the", "and", etc, whereas advertisements (especially primitive spam) use a different subset of atoms with different frequencies. And by using a CART system like xgboost for classification we build a set of decision rules to differentiate how mbed english and spam english texts are *constructed*.

# Results

Training SpamBlaster on notebook/forum posts for the past 10 days and validating with stratified k-fold (7 folds) yields 100% AUC score. Note, nearly all of the spam posts are rudimentary at best. Consequently, if we retrain with more advanced spam samples we expect to see 90%-98% AUC.
