"""One-class classifiers"""
from abc import abstractmethod
import numpy as np
from miniball import get_bounding_ball
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors

class OneClassClassifier():
    """Abstract base class for one-class classification"""
    
    def __init__(self):
        pass
           
    
    @abstractmethod
    def train(self, positive_patterns):
        """Train the classifier (build the model)
        
        Parameters
        ----------
        positive_patterns: nparray of float (N,D)
            The positive instances used to train the classifier, where N is
            the number of instances and D the dimension of the feature space
        """
        
    @abstractmethod
    def predict(self, test_patterns):
        """Test if the patterns belong to the given class 
        
        Parameters
        ----------
        test_patterns: nparray of float (N,D)
            The patterns to test, where N is the number of instances and 
            D the dimension of the feature space.
        
        Returns
        -------
        predictions: ndarray of int
            Indices of the test patterns that belong to the given class
        """    

class SVDD(OneClassClassifier):
    """Support Vector Data Description classifier"""
    
    def train(self, positive_patterns):
        #Find the smallest bounding ball enclosing the train data
        self._center, r2 = get_bounding_ball(positive_patterns)
        self._radius = np.sqrt(r2)
        
        #Reshape centre as row vector
        self._center = np.reshape(self._center, (1, len(self._center)))
        
    def predict(self, test_patterns):
        #Compute the distance between each test pattern to the ball centre
        dists = distance_matrix(self._center, test_patterns)
        
        #Threshold the distance at the radius value
        predictions = np.argwhere(dists.flatten() <= self._radius)
        
        return predictions
    
class NND(OneClassClassifier):
    """Nearest Neighbour Description"""
    
    def train(self, positive_patterns):
        self._positive_patterns = positive_patterns
        
    def predict(self, test_patterns):
        
        #For each test pattern find the nearest positive pattern (1st nn)
        #and the corresponding distance
        nnsearcher = NearestNeighbors(n_neighbors=1)
        nnsearcher.fit(self._positive_patterns)
        dists_test_patterns_to_1st_nn, idxs_1st_nn =\
            nnsearcher.kneighbors(test_patterns, return_distance = True)
        _1st_nn = self._positive_patterns[idxs_1st_nn[:,0],:] 
        dists_test_patterns_to_1st_nn = dists_test_patterns_to_1st_nn.flatten()
        
        #------------------------------------------------------------------
        #For each 1st nn find the nearest positive pattern (2nd nn) and the
        #corresponding distance
        
        del(nnsearcher)
        nnsearcher = NearestNeighbors(n_neighbors=2)
        nnsearcher.fit(self._positive_patterns)
        
        dists_1st_nn_to_2nd_nn, idxs_2nd_nn =\
            nnsearcher.kneighbors(_1st_nn, return_distance = True)
        
        #Discard the first nns, which are the query points themselves
        dists_1st_nn_to_2nd_nn = dists_1st_nn_to_2nd_nn[:,1]
        dists_1st_nn_to_2nd_nn = dists_1st_nn_to_2nd_nn.flatten()
        idxs_2nd_nn = idxs_2nd_nn[:,1]
        
        #------------------------------------------------------------------
        
        #Return the indices of the patterns for which the distance 1st-nn to
        #2nd-nn is less than that between test pattern and 1st-nn
        predictions = np.argwhere(dists_test_patterns_to_1st_nn < dists_1st_nn_to_2nd_nn)
            
        return predictions
