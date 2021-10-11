import cenotaph.classification.classifiers as cl
import numpy as np

train_patterns = np.array([[0., 0., 0.], [0., .5, 0.], [1., 1., .5]])
train_labels = ['Hearts', 'Clubs', 'Spades']
test_patterns = np.array([[0., 0., 0.1], [0., .5, 0.1]])

knnclassifier = cl.KNNClassifier()
predicted_labels = knnclassifier.predict(train_patterns, train_labels, test_patterns)
print(predicted_labels)

train_patterns = np.array([[0., 0., 0.1], [0., .5, 0.], [1., 1., .5]])
train_labels = ['Hearts', 'Clubs', 'Spades']
predicted_labels = knnclassifier.predict(train_patterns, train_labels, test_patterns)
print(predicted_labels)
