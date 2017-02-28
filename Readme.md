# Spam Blaster
 Author: Michael Bartling

# Overview

Spam Blaster is a simple attempt to thwart spam using basic data mining techniques. The idea is to train some bag of words classifier to estimate the probability a sample was taken from some language distribution. The feature vectors are sentences in a document and the document is classified as spam/not-spam by ensembling the individual sentence predictions.

# Notes on Spam

Currently, most of the spam is taken directly from the mbed forums. However, this spam is not very diverse and often appears in the form WORD1 WORD1 WORD1 RANDOMWORD WORD1 WORD1 WORD1. Although this makes it easy to detect using signature based approaches, signatures suffer from over specialization and hand crafting. Therefore, I extend the spam feature set to include samples taken from http://untroubled.org/spam/
