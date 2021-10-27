"""One-class classifiers"""
from abc import abstractmethod
import numpy as np
from miniball import get_bounding_ball
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EllipticEnvelope as OneClassEllipticEnvelope
from sklearn.svm import OneClassSVM

class OneClassClassifier():
    """Abstract base class for one-class classification"""
    
    def __init__(self):
        pass
           
    
    @abstractmethod
    def train(self, inlier_patterns):
        """Train the classifier (build the model)
        
        Parameters
        ----------
        inlier_patterns: nparray of float (N,D)
            The inlier instances used to train the classifier, where N is
            the number of instances and D the dimension of the feature space.
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
            Indices of the test patterns that belong to the given class (inliers).
            The others are considered outliers.
        """    

class SklearnStandard(OneClassClassifier):
    """Base class for one-class classifier based on fit/predict sklearn
    models"""
    
    @abstractmethod
    def __init__(self):
        pass
        
    def train(self, inlier_patterns):
        self._model.fit(X = inlier_patterns)
        
    def predict(self, test_patterns):
        results = self._model.predict(X = test_patterns)
        predictions = np.argwhere(results == 1)
        
        return predictions    

class EllipticEnvelope(SklearnStandard):
    """One-class classifier based on elliptic envelope"""
    
    def __init__(self, contamination = 0.05):
        """
        Parameters    
        -------------
        contamination: float
            The expected proportion of outliers in the training data. Must be in 
            the (0, 0.5] interval.         
        """     
        self._model = OneClassEllipticEnvelope(contamination = contamination, 
                                               random_state = 0)
        
class SVM(SklearnStandard):
    """One-class support vector classifier"""
    
    def __init__(self, gamma = 'scale', nu = 0.5):
        """
        Parameters    
        -------------
        gamma: {'scale', 'auto'} or float, default='scale'
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma='scale' 
            is passed then it uses 1 / (n_features * X.var()) as value of gamma,
            if 'auto', uses 1 / n_features.
        nu: float 
            An upper bound on the fraction of training errors and a lower bound 
            of the fraction of support vectors. Should be in the interval (0, 1].         
        """
        self._model = OneClassSVM(gamma = gamma, nu = nu)
        
class SVDD(OneClassClassifier):
    """Support Vector Data Description classifier"""
    
    def train(self, inlier_patterns):
        #Find the smallest bounding ball enclosing the train data
        self._center, r2 = get_bounding_ball(inlier_patterns)
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
    """Nearest Neighbour Description (see [1, Sec. 4.2.2.4])
    
    References
    ----------
    [1] Khan, S.S., Madden, M.G.
        One-class classification: Taxonomy of study and review of techniques
        (2014) Knowledge Engineering Review, 29 (3), pp. 345-374. 
    """
    
    def __init__(self, k=1):
        """
        Parameters
        ----------
        k: int
            Define the first neighbour (FPN) as the (positive) train pattern 
            closest to the test pattern. The parameter defines the number of
            FPN nearest neighbours over which the average distance
            is computed.
        """
        self._k = k
    
    def train(self, inlier_patterns):
        self._inlier_patterns = inlier_patterns
        
    def predict(self, test_patterns):
        
        #For each test pattern find the nearest positive pattern (1st nn)
        #and the corresponding distance
        nnsearcher = NearestNeighbors(n_neighbors=1)
        nnsearcher.fit(self._inlier_patterns)
        dists_test_patterns_to_1st_nn, idxs_1st_nn =\
            nnsearcher.kneighbors(test_patterns, return_distance = True)
        _1st_nn = self._inlier_patterns[idxs_1st_nn[:,0],:] 
        dists_test_patterns_to_1st_nn = dists_test_patterns_to_1st_nn.flatten()
        
        #------------------------------------------------------------------
        #For each 1st nn find the K nearest positive patterns (2nd nn) and the
        #corresponding distances
        
        del(nnsearcher)
        nnsearcher = NearestNeighbors(n_neighbors=self._k + 1)
        nnsearcher.fit(self._inlier_patterns)
        
        dists_1st_nn_to_knn, idxs_knn =\
            nnsearcher.kneighbors(_1st_nn, return_distance = True)
        
        #Discard the first k nns, which are the query points themselves
        dists_1st_nn_to_knn = dists_1st_nn_to_knn[:,1::]
        
        #Average the distance over the first k nns
        avg_dist_1st_nn_to_knn = np.mean(dists_1st_nn_to_knn, axis = 1)
        avg_dist_1st_nn_to_knn = avg_dist_1st_nn_to_knn.flatten()
        #------------------------------------------------------------------
        
        #Return the indices of the patterns for which the distance 1st-nn to
        #2nd-nn is less than that between test pattern and 1st-nn
        predictions = np.argwhere(dists_test_patterns_to_1st_nn < avg_dist_1st_nn_to_knn)
            
        return predictions
    
class NNPC(OneClassClassifier):
    """Nearest Neighbour Positive Class (see [1, Sec. 4.2.2.4])
    
    References
    ----------
    [1] Khan, S.S., Madden, M.G.
        One-class classification: Taxonomy of study and review of techniques
        (2014) Knowledge Engineering Review, 29 (3), pp. 345-374. 
    """  
    
    def __init__(self, mode='max'):
        """
        Parameters
        ----------
        mode: str
            A string indicating the classification strategy. Can be:
                'max' -> A target is considered in the class if the distance
                         from the nearest positive pattern is less than the 
                         maximum pairwise distance among the nearest 
                         positive patterns [1, Eq. 2]
        """
        self._mode = mode
        
    def train(self, inlier_patterns):
        self._inlier_patterns = inlier_patterns
        
        #------ Pairwise distances between nearest positive patterns ------
        nnsearcher = NearestNeighbors(n_neighbors=2)
        nnsearcher.fit(self._inlier_patterns)       
        pdists, _ = nnsearcher.kneighbors(self._inlier_patterns, 
                                          return_distance = True)        
        pdists = pdists[:,1].flatten()
        #------------------------------------------------------------------
        
        if self._mode == 'max':
            self._threshold = np.max(pdists)
        else:
            raise Exception(f'Unsupported mode *{mode}*') 
        
    def predict(self, test_patterns):
        
        #For each test pattern find the distance to the nearest positive pattern
        nnsearcher = NearestNeighbors(n_neighbors=1)
        nnsearcher.fit(self._inlier_patterns)
        target_dists, _ = nnsearcher.kneighbors(test_patterns, 
                                                return_distance = True)
        target_dists = target_dists.flatten()
        
        #Consider 'in class' the targets whose distance to the nearest positive
        #pattern in the train group is less than the threshold
        predictions = np.argwhere(target_dists < self._threshold)
            
        return predictions        
            