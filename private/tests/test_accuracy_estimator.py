import numpy as np
import pandas as pd

from cenotaph.classification.accuracy_estimation import AccuracyEstimator
from cenotaph.classification.accuracy_estimation import FiguresOfMerit
from cenotaph.classification.accuracy_estimation import FullSampling
from cenotaph.classification.accuracy_estimation import LeaveOneOut
from cenotaph.classification.accuracy_estimation import StratifiedSampling
import cenotaph.classification.classifiers as cl

labels = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C',]
train_ratio = 0.25
num_splits = 4
stratified_sampling = StratifiedSampling(None, train_ratio, num_splits, 
                                         labels, None)
splits = stratified_sampling._get_splits(len(labels))
print('Stratified sampling')
print(splits)

full_sampling = FullSampling(None, train_ratio, num_splits, None)
splits = full_sampling._get_splits(len(labels))
print('Full sampling')
print(splits)

leave_one_out = LeaveOneOut(None)
splits = leave_one_out._get_splits(len(labels))
print('Leave-one-out')
print(splits)


#Import the Iris dataset
source = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(source, header=None)

#Get the patterns and ground truth
patterns = np.array(df.iloc[:,0:4])
gt = (df.iloc[:,4]).tolist() 

#Define a classifier
classifier = cl.KNNClassifier()

#***************************************
#*********** Test direct ***************
#***************************************
estimation_mode = 'direct'
n_splits = 15
splits = np.random.choice(a=[True,False], size=(patterns.shape[0],n_splits))
acc_est = AccuracyEstimator.create_estimator(classifier,
                                             estimation_mode, 
                                             splits=splits)

#Get the accuracy and print the results
avg, std, acc = acc_est.get_accuracy(patterns, gt)
print("Average accuracy (%s) = %4.2f \u00B1 %4.3f" % (estimation_mode, avg, std))

#***************************************
#****** Test stratified sampling *******
#***************************************
#Define the accuracy estimator
estimation_mode = 'stratified'
train_ratio = 0.25
n_splits = 15
acc_est = AccuracyEstimator.create_estimator(classifier,
                                             estimation_mode=estimation_mode, 
                                             train_ratio=train_ratio,
                                             num_splits=n_splits)

#Get the accuracy and print the results
avg, std, acc = acc_est.get_accuracy(patterns, gt)
print("Average accuracy (%s) = %4.2f \u00B1 %4.3f" % (estimation_mode, avg, std))

#***************************************
#********** Test full sampling *********
#***************************************
#Define the accuracy estimator
estimation_mode = 'full'
acc_est = AccuracyEstimator.create_estimator(classifier,
                                             estimation_mode=estimation_mode, 
                                             train_ratio=train_ratio,
                                             num_splits=n_splits)

#Get the accuracy and print the results
avg, std, acc = acc_est.get_accuracy(patterns, gt)
print("Average accuracy (%s) = %4.2f \u00B1 %4.3f" % (estimation_mode, avg, std))

#*******************************************
#**Test full sampling - splits stored (1) **
#*******************************************
#Define the accuracy estimator
estimation_mode = 'full'
acc_est = AccuracyEstimator.create_estimator(classifier,
                                             estimation_mode=estimation_mode, 
                                             train_ratio=train_ratio,
                                             num_splits=n_splits,
                                             frozen_splits='./splits.txt')

#Get the accuracy and print the results
avg, std, acc = acc_est.get_accuracy(patterns, gt)
print("Average accuracy (%s) = %4.2f \u00B1 %4.3f" % (estimation_mode, avg, std))

#*******************************************
#**Test full sampling - splits stored (2) **
#*******************************************
#Define the accuracy estimator
estimation_mode = 'full'
acc_est = AccuracyEstimator.create_estimator(classifier,
                                             estimation_mode=estimation_mode, 
                                             train_ratio=train_ratio,
                                             num_splits=n_splits,
                                             frozen_splits='./splits.txt')

#Get the accuracy and print the results
avg, std, acc = acc_est.get_accuracy(patterns, gt)
print("Average accuracy (%s) = %4.2f \u00B1 %4.3f" % (estimation_mode, avg, std))

#Test figures of merit
fom = acc_est.get_figures_of_merit(patterns, gt)
destination = './fom.txt'
fom.save(destination)
fom_loaded = FiguresOfMerit.load(destination)

