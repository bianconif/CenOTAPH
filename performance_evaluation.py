import os.path

import numpy as np

from cenotaph.basics.base_classes import Image
from cenotaph.classification.accuracy_estimation import FiguresOfMerit
from cenotaph.basics.generic_functions import get_files_and_folders
from cenotaph.basics.base_classes import ImageProcessingPipeline

class PerformanceEvaluator:
    """Container class for performance evaluation of an image processing
    pipeline"""
    
    def __init__(self, ipp, classifier, acc_est, dataset, feature_file,
                 classification_file):
        """Default constructor
        
        Parameters
        ----------
        ipp : ImageProcessingPipeline
            The image processing pipeline the performance of which we want to 
            estimate
        classifier : Classifier
            The classifier used for the estimation
        acc_est : AccuracyEstimator
            The accuracy estimator
        dataset : str
            A pointer to the root folder of the image dataset. The dataset should
            be organised as follows: the top-level subfolders represent the
            classes; all the files contained in each of the top-level
            subfolders and the lower level subfolders belong to that class. The
            following example shows a tree structure with two classes:
            A (samples a1.png, a2.png, a3.png and a4.png) and B (b1.png and
            b2.png)
            
            [root]
            |
            +--- [class-A]
                     |
                     +--- a1.png
                     +--- a2.png
                     |
                     +--- [class-A1]
                               |     
                               +--- a3.png
                               +--- a4.png
            +--- [class-B]
                     +--- b1.png
                     +--- b2.png
        feature_file : str
            Absolute or relative path to the file where the features are stored. 
            If the file exists the features are read from there and
            not computed; otherwise they are computed from scratch and stored
            in feature_file. To force the features to be recomputed just delete
            feature_file.
        classification_file : str
            Absolute or relative path to the file where the classification are 
            stored. If the file exists the classification results are read from 
            there and not computed; otherwise they are computed from scratch and stored
            in feature_file. To force the features to be recomputed just delete
            feature_file.
        """
        
        if not os.path.isdir(dataset):
            raise Exception("Couldn't open dataset " + dataset)
        
        self._ipp = ipp
        self._classifier = classifier
        self._acc_est = acc_est
        self._dataset = dataset
        self._feature_file = feature_file
        self._classification_file = classification_file
                
    def get_performance(self):
        """Performance of the system
        
        Returns
        -------
        performance : FiguresOfMerit
            The performance of the system
        """
        
        #Check if the results are stored, otherwise compute and store them
        performance = None
        if os.path.exists(self._classification_file):
            performance = FiguresOfMerit.load(self._classification_file)
        else:
            performance = self._compute_performance()
            performance.save(self._classification_file)
        return performance
        
    def _compute_performance(self):
        """Computes the classification results
        
        Returns
        -------
        performance : FiguresOfMerit
            The performance of the system
        """
        
        performance = None
        
        #Check if the features are stored somewhere. If so load, otherwise
        #compute nd store them
        if os.path.exists(self._feature_file):
            features = ImageProcessingPipeline.load_features(self._feature_file)
        else:
            #Get the list of the images to process
            img_files, _ = get_files_and_folders(self._dataset)
            
            #Process the first image to get the feature length
            num_features = len(self._ipp.\
                compute_features(Image(img_files[0])))
            
            #Initialise the feature matrix
            features = np.zeros((len(img_files),num_features))
            
            #Compute and store the features
            for i in range(len(img_files)):
                features[i,:] = self._ipp.compute_features(Image(img_files[i]))         
            ImageProcessingPipeline.save_features(features, self._feature_file)
        
        #Retrieve the ground truth labels
        _ , ground_truth = get_files_and_folders(self._dataset)
        
        #Estimate the performance        
        performance = self._acc_est.get_figures_of_merit(features, ground_truth)
        
        return performance
