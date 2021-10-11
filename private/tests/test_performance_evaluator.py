from cenotaph.basics.base_classes import ImageProcessingPipeline
from cenotaph.performance_evaluation import PerformanceEvaluator
import cenotaph.colour_descriptors as cd
import cenotaph.preprocessing.colour.colour_constancy as cc
import cenotaph.classification.classifiers as cl
import cenotaph.classification.accuracy_estimation as ae

#************************************
#Set up the image processing pipeline
#************************************
#Image descriptors
mean = cd.Mean()
mean_std = cd.MeanStd()
image_descriptors = [mean]

#Preprocessors
chroma = cc.Chroma()
gw = cc.GreyWorld()
preprocessors = [gw]

#Pipeline
ipp = ImageProcessingPipeline(image_descriptors, preprocessors)
#************************************
#************************************
#************************************

#Define the classifier
nn = cl.KNNClassifier()

#Define the accuracy estimator
fs = ae.FullSampling(nn, train_ratio = 0.25, 
                     num_splits = 15, frozen_splits=None)

#Define the repositories
source = 'G:\Data\Disk_I\LACIE\ImageDatasets\Texture\KTH-TIPS'
feats_repo = './features.npy'
perf_repo = './performance.txt'

#Create the evaluator and get the performance
perf_evaluator = PerformanceEvaluator(ipp, nn, fs, source, 
                                      feats_repo, perf_repo)
performance = perf_evaluator.get_performance()