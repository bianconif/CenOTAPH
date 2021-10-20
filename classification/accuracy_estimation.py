from abc import ABC, abstractmethod
import os
import pickle

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

class FiguresOfMerit:
    """A container for the figures of merit used to evaluate the accuracy
    of a classifier"""
    
    def __init__(self, predicted_labels, correct_labels):
        """Default constructor
        
        Parameters
        ----------
        predicted_labels : list of list (str)
            A list containing P lists, where P is the number of subdivisions
            into train and test set (number of problems). 
            Each list contains the labels (str) predicted by the classifier
            for the p-th classification problem
        correct_labels : list of list (str) 
            A list containing P lists, where P is the number of subdivisions
            into train and test set (number of problems). 
            Each list contains the correct labels/ground truth (str) for the 
            p-th classification problem
        """
        
        #Make sure predicted and correct labels are the same length
        _labels_same_length = True
        if len(predicted_labels) != len(correct_labels):
            _labels_same_length = False
            if _labels_same_length:
                for i in range(len(predicted_labels)):
                    if len(predicted_labels[i]) != len(correct_labels[i]):
                        _labels_same_length = False
        
        if not _labels_same_length:
            raise Exception('Predicted and correct labels are not the same length')
            
      
        self._predicted_labels = predicted_labels
        self._correct_labels = correct_labels
        self._accuracy = None
        
        #Set the flags
        self._accuracy_computed = False
    
    def get_sensitivity_and_specificity(self, 
                                        label_positive='P', 
                                        label_negative='N'):
        """Sensitivity and specificity of a binary classifier
        
        Parameters
        ----------
        label_positive : int or str
            Label used for identifying the positive cases
        label_negative : int or str
            Label used for identifying the negative cases
            
        Returns
        -------
        avg_sensitivity : float
            The sensitivity of the binary classifier averaged over the
            classification problems
        std_sensitivity : float
            The standard deviation of the sensitivity of the binary classifier 
            over the classification problems
        avg_specificity : float
            The specificity of the binary classifier averaged over the
            classification problems
        std_specificity : float
            The standard deviation of the specificity of the binary classifier 
            over the classification problems
        sensitivity : list of float
            The sensitivity by classification problem
        specificity : list of float
            The specificity by classification problem
        """
        
        #Check the input
        flat_predicted_labels = [item for sublist in self._predicted_labels for
                                 item in sublist]
        flat_correct_labels   = [item for sublist in self._correct_labels for
                                 item in sublist]     
        classes = list(set(flat_predicted_labels + flat_correct_labels))
        classes_match_labels_option_1 = (classes[0] == label_positive) and\
                                        (classes[1] == label_negative)
        classes_match_labels_option_2 = (classes[1] == label_positive) and\
                                        (classes[0] == label_negative)        
        
        #Raise an exception if the number of classes is not two or if the 
        #class labels do not match labels passed as input parameters
        #(label_positive and label_negative)
        if not len(classes) == 2:
            raise Exception('Not a binary classification problem')
        elif not (classes_match_labels_option_1 or
                  classes_match_labels_option_2):
            raise Exception('The class labels do not match labels passed as'
                            'input parameters') 
        
        #Iterate through the classification problems
        sensitivity = list()
        specificity = list()
        for p, _ in enumerate(self._correct_labels):        
        
            #Compute the elements of the 2x2 confusion matrix
            predicted_positives = [i for i, x in enumerate(self._predicted_labels[p]) 
                                   if x == label_positive]
            predicted_negatives = [i for i, x in enumerate(self._predicted_labels[p]) 
                                   if x == label_negative]
            correct_positives   = [i for i, x in enumerate(self._correct_labels[p]) 
                                   if x == label_positive]
            correct_negatives   = [i for i, x in enumerate(self._correct_labels[p]) 
                                   if x == label_negative]
            
            #True positives (tp), true negatives (tn), false positives (fp)
            #and false negatives (fn)
            tp = set(predicted_positives).intersection(set(correct_positives))
            tn = set(predicted_negatives).intersection(set(correct_negatives))
            fp = set(predicted_positives).intersection(set(correct_negatives))
            fn = set(predicted_negatives).intersection(set(correct_positives))
                        
            sensitivity.append(len(tp)/(len(tp) + len(fn)))
            specificity.append(len(tn)/(len(tn) + len(fp)))
        
        return np.mean(sensitivity), np.std(sensitivity),\
               np.mean(specificity), np.std(specificity),\
               sensitivity, specificity
    
        
    def get_accuracy(self):
        """Returns the accuracy by problem
        
        Returns
        -------
        avg : float
            The average accuracy
        std : float
            The accuracy standard deviation
        self._accuracy : ndarray (float)
            The accuracy by problem (array of P elements)
        """
        
        if not self._accuracy_computed:
            self._compute_accuracy()
            
        return np.average(self._accuracy),\
               np.std(self._accuracy),\
               self._accuracy
    
    def _compute_accuracy(self):
        """Computes the accuracy by problem"""
        self._accuracy = list()
        
        #Iterate through the splits
        for p in range(len(self._predicted_labels)):
            matches = [(a == b) for a,b in zip(self._predicted_labels[p],
                                               self._correct_labels[p])]
            self._accuracy = np.append(self._accuracy,
                                       matches.count(True)/\
                                       len(self._correct_labels[p]))        
        self._accuracy = np.asarray(self._accuracy)      
        self._accuracy_computed = True
        
    def save(self, out):
        """Saves the object on a file
        
        Parameters
        ----------
        out : str
            The destination file where the object is to be stored
        """

        destination = None
        try:
            destination = open(out,'wb')
            pickle.dump(self, destination)
        except:
            raise Exception('Something went wrong when trying to save')
        finally:
            if destination:
                destination.close()
             
    @staticmethod            
    def load(source):
        """Loads the object from a file
        
        Parameters
        ----------
        source : str
            The source file where the object is stored
            
        Returns
        -------
        fom : FiguresOfMerit
            The object
        """
        
        fom = None
        repo = None
        try:
            repo = open(source,'rb')
            fom = pickle.load(repo)
        except:
            raise Exception('Something went wrong when trying to load')
        finally:
            if repo:
                repo.close() 
        
        return fom

class AccuracyEstimator(ABC):
    """Base class for accuracy estimation"""
    
    @staticmethod
    def create_estimator(classifier, estimation_mode, **options):
        """Factory function for creating concrete estimators
        
        Parameters
        ----------
        classifier : Classifier
            A concrete instance of an empty classifier to be used for the
            classification
        estimation_mode : str
            The accuracy estimation procedure. Possible values are:
                'direct'        -> splits into train and test are passed 
                                   explicitly
                'full'          -> full sampling
                'leave-one-out' -> leave-one-out
                'stratified' -> stratified sampling
        splits (required if estimation_mode = 'direct', unused otherwise) 
            : ndarray of int
            See Direct.__init__() for the meaning of values
        train_ratio : float [0.0,1.0] (default=0.5)
            The fraction of samples that go to the train set
            Required if estimation mode = 'full' or 'stratified', unused 
            otherwise
        num_splits : int (default=10)
            The number of splits into train and test set
            Required if estimation mode = 'full' or 'stratified', unused 
            otherwise
        ground_truth : list of int or str
            The class labels (ground truth). Required if estimation_mode =
            'stratified', unused otherwise
        frozen_splits : str
            The full path to the file where the splits are to be stored. If
            the file is empty or non-existent the splits are created on the first
            call and stored. On subsequent calls the function reads the splits
            stored withouth recomputing them. This option ensures repeatability
            of the results between subsequent calls on the same dataset or
            different datasets with the same size.
            
        Returns
        -------
        accuracy_estimator : instance of AccuracyEstimator
            The accuracy estimator
        """
        
        #Initialise the output
        accuracy_estimator = None
        
        if estimation_mode == 'direct':
            if 'splits' not in options.keys():
                raise Exception('Splits into train and test set are required')       
            accuracy_estimator = Direct(classifier, options['splits'])
        elif estimation_mode == 'full':
            accuracy_estimator = FullSampling(classifier, 
                                              options['train_ratio'],
                                              options['num_splits'],
                                              options['frozen_splits'])
        elif estimation_mode == 'leave-one-out':
            accuracy_estimator = LeaveOneOut(classifier)
        elif estimation_mode == 'stratified':
            train_ratio, num_splits, frozen_splits = AccuracyEstimator._unpack_params(options)
            accuracy_estimator = StratifiedSampling(classifier, 
                                              options['train_ratio'],
                                              options['num_splits'],
                                              options['ground_truth'],
                                              options['frozen_splits']) 
        else:
            raise Exception('Accuracy estimation mode not supported')
        
        return accuracy_estimator
  
    def __repr__(self):
        return self.__class__.__name__                       
    
    def __init__(self, classifier):
        """The default constructor
        
        Parameters
        ----------
        classifier : Classifier
            The empty classifier to be used for accuracy estimation
        For optional parameters see create_estimator()
        """
        
        self._classifier = classifier
        
        #Initialise patterns and ground truth
        self._patterns = None
        self._ground_truth = None
        
        #Initialise the splits repo
        self._frozen_splits = None
        
        #Set the flags    
        self._splits_computed = False

    def get_accuracy(self, patterns, ground_truth):
        """Get the estimated accuracy
        
        Parameters
        ----------
        patterns : ndarray (int, float)
            The patterns dataset (N,F). Rows correspond to observations and
            columns to features.
        ground_truth : array-like (int, str)
            The correct labels (N) of each pattern
        
        Returns
        -------
        avg : float
            The overall accuracy value averaged over the splits
        std : float
            The standard deviation of the accuracy value over the splits
        accuracy : ndarray
            The accuracy for each split
        """
        
        #Delegate to get_figures_of_merit
        fom = self.get_figures_of_merit(patterns, ground_truth)      
        return fom.get_accuracy()

    def get_figures_of_merit(self, patterns, ground_truth):
        """Returns the figures of merit
        
        Parameters
        ----------
        patterns : ndarray (int, float)
            The patterns dataset (N,F). Rows correspond to observations and
            columns to features.
        ground_truth : array-like (int, str)
            The correct labels (N) of each pattern
        
        Returns
        -------
        performance : FiguresOfMerit
            The performance of the system
        """
        
        if not self._splits_computed:
            self._get_splits(patterns.shape[0])
            self._splits_computed = True            
            
        #Iterate through the splits and perform the classification
        predicted_labels = list()
        correct_labels = list()
        for s in range(self._splits.shape[1]):
            #Get the train and test indices
            train_idxs = np.where(self._splits[:,s] == 1)[0]
            test_idxs = np.where(self._splits[:,s] == 0)[0]
            
            #Get the train, test patterns and labels
            train_patterns = patterns[train_idxs,:]
            test_patterns = patterns[test_idxs,:]
            train_labels = [ground_truth[i] for i in train_idxs]
            test_labels = [ground_truth[i] for i in test_idxs]
            
            #Do the classification
            predicted = self._classifier.predict(train_patterns, 
                                                 train_labels, 
                                                 test_patterns)
            
            #Append the results
            predicted_labels.append(predicted.tolist())
            correct_labels.append(test_labels)
        
        #Create end return a FiguresOfMerit object    
        performance = FiguresOfMerit(predicted_labels, correct_labels)  
        return performance

    def _get_splits(self, num_patterns):
        """Read or compute the splits into train and test set
        
        Parameters
        ----------
        num_patterns : int
            The number of patterns (observations in the dataset)
        """
        
        if self._frozen_splits:
            
            #Check if file exists first
            if os.path.exists(self._frozen_splits):
                #In this case just read the data
                self._splits = np.loadtxt(self._frozen_splits)
            else:
                #Otherwise compute the splits and store them
                self._compute_splits(num_patterns)
                np.savetxt(self._frozen_splits, self._splits, fmt='%d')
        else:      
            #If save splits not required just compute them
            self._compute_splits(num_patterns)
            
        return self._splits

    @abstractmethod
    def _compute_splits(self, num_patterns):
        """Compute the splits into train and test set. The polymorphic method
        to be implemented by the subclasses
        
        Parameters
        ----------
        num_patterns : int
            The number of patterns (observations in the dataset)
        """        

                
class Direct(AccuracyEstimator):
    """Train and test splits are defined explicitly"""
    
    def __init__(self, classifier, splits):
        """
        Parameters
        ----------
        classifier : Classifier
            The empty classifier to be used for accuracy estimation
        splits : ndarray of int
            The splits into train and test set (N,S). Rows correspond to
            observations and columns to splits. The value of each entry 
            respectively indicates: 
                1               -> The corresponding pattern is a train pattern
                0               -> The corresponding pattern is a test pattern
                Any other value -> The corresponding pattern is not used
        """        
        super().__init__(classifier)
        self._splits = splits
        
    def _compute_splits(self, num_patterns):
        """Do nothing, splits are already passed to the constructor"""
        return
    
class LeaveOneOut(AccuracyEstimator):
    """Leave one out"""
    
    def __init__(self, classifier):
        super().__init__(classifier)
       
    def _compute_splits(self, num_patterns):
        #Create a matrix with all entries = 1 and main diagonal entries = 0
        splits = np.ones((num_patterns, num_patterns), dtype = int)
        np.fill_diagonal(splits, val = 0)
        self._splits = splits
        
    
class FullSampling(AccuracyEstimator):
    """Accuracy estimation via full sampling"""
       
    def __init__(self, classifier, train_ratio, num_splits, frozen_splits):
        """ The constructor
        Parameters
        ----------
        classifier : Classifier
            The empty classifier to be used for accuracy estimation
        train_ratio : float [0.0,1.0]
            The fraction of samples of each class that go to the train set
        num_splits : int 
            The number of splits into train and test set
        frozen_splits : str
            The full path to the file where the splits are to be stored. If
            the file is empty or non-existent the splits are created on the first
            call and stored. On subsequent calls the function reads the splits
            stored withouth recomputing them. This option ensures repeatability
            of the results between subsequent calls on the same dataset or
            different datasets with the same size. Pass None if you don't want the
            splits frozen.
        """
        super().__init__(classifier)
        self._train_ratio = train_ratio
        self._num_splits = num_splits
        self._frozen_splits = frozen_splits
              
        #Initialise the splits with an empty array
        self._splits = np.array([], dtype = int)         
        
        #Define the splitter (full sampling)
        self._splitter = ShuffleSplit(self._num_splits, 
                                      1.0 - self._train_ratio) 
        
    def _compute_splits(self, num_patterns):
        """Compute the splits into train and test set 
        
        Parameters
        ----------
        num_patterns : int
            The number of patterns (observations in the dataset)
        """         
        
        self._splits = np.zeros((num_patterns,self._num_splits))
        
        problem = 0
        foo_patterns = np.zeros((num_patterns,1))
        for train_index, test_index in self._splitter.split(foo_patterns,
                                                            self._ground_truth):
            #Set True (1) when the pattern goes to the train set
            self._splits[train_index,problem] = 1
            problem+=1   
            
    def __repr__(self):
        train_ratio_str = '{}'.format(self._train_ratio)
        train_ratio_str = train_ratio_str.replace('.', '')
        return '{}-{}-{}'.format(self.__class__.__name__,
                                 self._num_splits,
                                 train_ratio_str)    
        
class StratifiedSampling(FullSampling):
    """Accuracy estimation via stratified sampling"""
    
    def __init__(self, classifier, train_ratio, num_splits, ground_truth,
                 frozen_splits):
        """
        Parameters
        ----------
        classifier : Classifier
            The empty classifier to be used for accuracy estimation
        train_ratio : float [0.0,1.0]
            The fraction of samples of each class that go to the train set
        num_splits : int 
            The number of splits into train and test set
        ground_truth : list of int or str
            The class labels
        frozen_splits : str
            The full path to the file where the splits are to be stored. If
            the file is empty or non-existent the splits are created on the first
            call and stored. On subsequent calls the function reads the splits
            stored withouth recomputing them. This option ensures repeatability
            of the results between subsequent calls on the same dataset or
            different datasets with the same size. Pass None if you don't want the
            splits frozen.
        """
        super().__init__(classifier, train_ratio, num_splits, frozen_splits)
        self._ground_truth = ground_truth
        
        #Set the splitter (stratified sampling)
        self._splitter = StratifiedShuffleSplit(self._num_splits, 
                                                1.0 - self._train_ratio)        

                
   
        
       
         
            
        
    
      