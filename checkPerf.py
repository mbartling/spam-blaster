import glob
import spamblaster
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

notSpamFiles = glob.glob("data/not-spam/*.txt")
areSpamFiles = glob.glob("data/spam/*.txt")

areSpamLabels = np.ones((len(areSpamFiles),))
notSpamLabels = np.zeros((len(notSpamFiles),))
labels = np.concatenate((areSpamLabels, notSpamLabels))
print labels.shape
mFiles = []
mFiles.extend(areSpamFiles)
mFiles.extend(notSpamFiles)
preds = []

for fname in mFiles:
    with open(fname) as fp:
        res = spamblaster.classifyBlobList(fp.read().split('.'))
        preds.append(np.mean(res[:,1]))


fpr, tpr, _ = roc_curve(list(labels), preds)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
