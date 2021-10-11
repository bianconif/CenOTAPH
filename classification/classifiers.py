from abc import ABC, abstractmethod

import sklearn.discriminant_analysis
import sklearn.ensemble
import sklearn.svm
import sklearn.tree
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from cenotaph.third_parties.doc_inherit import doc_inherit

class Classifier(ABC):
    """Base class for classifiers"""
    
    def __init__(self):
        """Default constructor"""
        
        self._train_patterns = None
        self._train_labels = None
        
        #The classification model is not set up yet
        self._model_set = False
                
        #The classifier is not trained yet
        self._trained = False
                
    def _set_train_data(self, train_patterns, train_labels):
        """Set the data for training
        
        Parameters
        ----------
        train_patterns : ndarray (int, float)
            An (N,F) matrix representing the train patterns, where N is the number
            of patterns (observations) and F the number of features
        train_labels : array-like (int, str)
            The train labels (N)
        """
        
        #Check if train labels and patterns are different, if so flag that the
        #classifier is not trained
        train_patterns_changed = False
        train_labels_changed = False
        num_features_changed = False
        if (self._train_patterns is not None) and (self._train_labels is not None):  
            
            #First check if the train labels and patterns have the same length as the
            #old ones
            train_patterns_same_length = (len(train_patterns) == 
                                          len(self._train_patterns))
            train_labels_same_length = (len(train_labels) == 
                                        len(self._train_labels))
            same_num_features = (train_patterns.shape[1] == 
                                 self._train_patterns.shape[1])
            
            #If they have the same length check if the content is the same too
            if train_patterns_same_length and same_num_features:
                train_patterns_changed = (self._train_patterns != train_patterns).any
            else:
                train_patterns_changed = True
            
            if train_labels_same_length and same_num_features:
                train_labels_changed = (self._train_labels != train_labels)
            else:
                train_labels_changed = True 

            
            if train_patterns_changed or train_labels_changed:
            
                self._trained = False
        
        #Debug code
        #if self._trained:
            #print('Already trained, non need to retrain')
        #else:
            #print('Train patterns/labels changed, need to retrain')
    
        self._train_patterns = train_patterns
        self._train_labels = train_labels
        

    
    def predict(self, train_patterns, train_labels, test_patterns):
        """Get the predicted labels -- abstract method, to be implemented in
        the derived classes
        
        Parameters
        ----------
        train_patterns : ndarray (int, float)
            An (N,F) matrix representing the train patterns, where N is the number
            of patterns (observations) and F the number of features
        train_labels : array-like (int, str)
            The train labels (N)
        test_labels : array-like (int, str)
            The train labels (M)
         
        Returns
        -------
        predicted_labels : array-like (int, str)
            The predicted labels (N)
        """
        
        #Set up the classification model
        if not self._model_set:
            self._set_model()
            self._model_set = True
             
        self._predicted_labels = None   
        self._set_train_data(train_patterns, train_labels)
        
        #Train the classifier if is not   
        if not self._trained:
            self._train()
        
        #Predict the response    
        self._predicted_labels = self._model.predict(test_patterns)
            
        return self._predicted_labels

    @abstractmethod
    def _set_model(self):
        """Sets up the classifier. Abstract method, to be implemented in the 
        subclasses"""
        
        pass
        
    def _train(self):
        """Train the classifier"""
        
        self._model.fit(self._train_patterns, self._train_labels)
        self._trained = True
        
    @staticmethod
    def factory(name, **kwargs):
        """The factory to generate classifiers
        
        Parameters
        ----------
        name : str
            The classifier name. Possible values are:
                "ClT"          -> Classification tree
                "KNN"          -> K-nearest-neighbours classifier
                "Linear"       -> Linear Discriminant Analysis
                "LinearSVM"    -> Linear Support Vector Classifier
                "NBGaussian"   -> Naive Bayes Gaussian classifier
                "RandomForest" -> Random forest classifier
                "RbfSVM"       -> Support Vector Machine with radial basis
                                  kernel function
            For details about usage see also the documentation of the 
            corresponding subclasses
        kwargs : dict
            The classifier-specific parameters. For possible values see the
            documentation of the corresponding subclasses
            
        Returns
        -------
        classifier : Classifier
            The classifier object
        """
        classifier = None
        if name == 'ClT':
            classifier = ClassificationTree(**kwargs)
        elif name == 'KNN':
            classifier = KNNClassifier(**kwargs)
        elif name == 'Linear':
                classifier = LinearClassifier(**kwargs)  
        elif name == 'LinearSVM':
                classifier = LinearSVMClassifier(**kwargs)                 
        elif name == 'NBGaussian':
            classifier = NBGaussianClassifier()
        elif name == 'RandomForest':
            classifier = RandomForestClassifier(**kwargs)
        elif name == 'RbfSVM':
            classifier = RbfSVMClassifier(**kwargs)
        else:
            raise Exception('Classifier not supported')
        
        return classifier
        
class NBClassifier(Classifier):
    """Abstract base class for Naive Bayes classifier"""
    
    def __init__(self, priors=None):
        """The constructor
    
        Parameters
        ----------
        priors : array-like, (n_classes,)
            Prior probabilities of the classes. If specified the priors are not
            adjusted according to the train data.
            """                
        super().__init__()  
        self._priors = priors
            
class NBGaussianClassifier(NBClassifier):
    """Gaussian Naive Bayes classifier"""
    
    def __init__(self, priors=None, var_smoothing=1e-09):
        """The constructor
    
        Parameters
        ----------
        priors : array-like, optional (n_classes,)
            A priori class probabilities. If specified the priors are not adjusted
            according to the train data.
        var_smoothing : float, optional (default=1e-9)
            Portion of the largest variance of all features that is added to 
            variances for calculation stability.
        """   
        super().__init__(priors)
        self._var_smoothing = var_smoothing
    
    def _set_model(self):
        #Set up the classification model
        self._model = GaussianNB(self._priors, self._var_smoothing)   

class KNNClassifier(Classifier):
    
    def __init__(self, n_neighbours=1, distance='L1'):
        """The constructor
        
        Parameters
        ----------
        n_neighbours : int
            The number of neighbours
        distance : str
            The type of distance to be used. Posssible values are:
                'L1'   -> Manhattan (cityblock) distance
                'L2'   -> Euclidean distance
        """                
        super().__init__()        
        
        self._n_neighbours = n_neighbours
        self._metric = 'minkowski'
        self._exp = None
        
        if distance == 'L1':
            self._exp = 1
        elif distance == 'L2':
            self._exp = 2
        else:
            raise Exception('Distance type not supported')
        
    def __repr__(self):
        return '{}NN-{}'.format(self._n_neighbours, self._exp)
             
    def _set_model(self):
        #Set up the classification model
        self._model = KNeighborsClassifier(n_neighbors=self._n_neighbours, 
                                           metric=self._metric,
                                           p=self._exp)
        
class RandomForestClassifier(Classifier):
    """Random forest classifier"""
    
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, random_state=0):
        """The constructor
        
        Parameters
        ----------
        n_estimators : int
            The number of trees in the forest
        criterion : str
            The function to measure the quality of a split. Possible values 
            are: 
                'entropy' -> Information gain
                'gini'    -> Gini impurity index
        max_depth : int
            The maximum depth of the tree
        min_sample_splits : int
            The minimum number of samples required to split an internal node
        min_sample_leaf : int
            The minimum number of samples required to be at a leaf node
        """ 
        super().__init__()
        
        self._n_estimators = n_estimators
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._random_state = random_state
        
    def _set_model(self):
        #Set up the classification model
        self._model = sklearn.ensemble.RandomForestClassifier(
            n_estimators = self._n_estimators,
            criterion = self._criterion,
            max_depth = self._max_depth, 
            min_samples_split = self._min_samples_split,
            min_samples_leaf = self._min_samples_leaf,
            random_state = self._random_state
        )    
    
class LinearClassifier(Classifier):
    """Linear classifier"""
    
    def __init__(self, solver='lsqr'):      
        """The constructor
    
        Parameters
        ----------
        solver : string
            Solver to use. Possible values are:
                'svd'   -> Singular value decomposition (default). Does not compute
                       the covariance matrix, therefore this solver is 
                       recommended for data with a large number of features.
                'lsqr'  -> Least squares estimation
                'eigen' -> Eigenvalue decomposition
        """ 
        super().__init__() 
        self._solver = solver
    
    def _set_model(self):
        #Set up the classification model
        self._model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
            solver = self._solver)   
        
class ClassificationTree(Classifier):
    """Classification tree"""
    
    def __init__(self, criterion='gini', splitter='best', max_depth=10,
                 min_samples_split=2, random_state=0):      
        """The constructor
    
        Parameters
        ----------
        criterion : str
            The function to measure the quality of a split. Possible values
            are:
                'entropy'   -> Information gain
                'gini'      -> Gini impurity index
        splitter : str 
            The strategy used to choose the split at each node. Possible
            values are:
                'best'      -> Best split
                'random'    -> Random split
        max_depth : int
            The maximum number of levels of the tree. If None, then nodes are 
            expanded until all leaves are pure or until all leaves contain 
            less than min_samples_split samples.
        min_samples_split : int
            The minimum number of samples required to split an internal node
        """ 
        super().__init__() 
        self._criterion = criterion
        self._splitter = splitter
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._random_state = random_state
        
    def _set_model(self):
        #Set up the classification model
        self._model = sklearn.tree.DecisionTreeClassifier(
            criterion = self._criterion,
            splitter = self._splitter,
            max_depth = self._max_depth,
            min_samples_split = self._min_samples_split,
            random_state = self._random_state
        ) 
        
class SVMClassifier(Classifier):
    """Abstract bas class for support vector classifiers"""
    
    def __init__(self, C=1.0, kernel='rbf'):
        """The constructor
    
        Parameters
        ----------
        C : float 
            The regularization parameter. The strength of the regularization
            is inversely proportional to C. Must be strictly positive. Penalty
            is computed by L2 norm.
        kernel : str
            the kernel type to be used in the algorithm. Possible values are:
                'linear'   -> linear kernel
                'poly'     -> polynomial kernel 
                'rbf'      -> radial basis kernel
                'sigmoid'  -> sigmoidal function
        """
        super().__init__()
        self._C = C
        self._kernel = kernel
        
    def __repr__(self):
        retval = self.__class__.__name__
        if self._grid_search:
            retval = retval + '-grid-search'
        return retval        
        
class RbfSVMClassifier(SVMClassifier):
    """Support Vector Machine Based on radial basis function"""
    
    def __init__(self, C=1.0, gamma='auto', grid_search=False, **kwargs):      
        """The constructor
    
        Parameters
        ----------
        C : float 
            The regularization parameter. The strength of the regularization
            is inversely proportional to C. Must be strictly positive. Penalty
            is computed by L2 norm.
        gamma : str or float
            Kernel coefficient. Possible values are:
                'scale' -> gamma = 1/(n_features * X.var())
                'auto'  -> gamma = 1/n_features
            If a float is given that is the value of gamma.
        grid_search : bool
            Whether to do grid search for hyperparameter optimisation. If True
            C and gamma values are overridden by the results of the grid
            search.
        C_grid : array-like (float)
            The search space for C. Required if grid_search = True, ignored
            otherwise
        gamma_grid : array-like (float)
            The search space for gamma. Required if grid_search = True, ignored
            otherwise
        """
        super().__init__(C = C, kernel = 'rbf')
        self._gamma = gamma
        self._grid_search = grid_search
        
        if self._grid_search:
            if 'C_grid' not in kwargs.keys():
                raise Exception("Search space for 'C' not given")
            if 'gamma_grid' not in kwargs.keys():
                raise Exception("Search space for 'gamma' not given")
            self._C_grid = kwargs['C_grid']
            self._gamma_grid = kwargs['gamma_grid']                
        
    def _do_grid_search(self, patterns, labels, C_grid, gamma_grid):
        """Hyperparameters grid search
        
        Parameters
        ----------
        patterns :  ndarray (int, float)
            An (N,F) matrix where N is the number of patterns (observations) 
            and F the number of features
        labels : array-like (int, str)
            The pattern labels (N)
        C_grid : array-like (float)
            The search space for the penalty parameter C
        gamma_grid : array-like (float)
            The search space for gamma
            
        Returns
        -------
        optimal_C, optimal_gamma : float
            The optimal values for C and gamma
        """
        
        svc = sklearn.svm.SVC()
        clf = GridSearchCV(svc, {'C' : C_grid, 'gamma' : gamma_grid})
        clf.fit(patterns, labels)
        
        return clf.best_estimator_.C, clf.best_estimator_.gamma
    
    #Override to allow grid search if required   
    @doc_inherit
    def predict(self, train_patterns, train_labels, test_patterns):
        #Do the parameter search if required
        if self._grid_search:
            optimal_C, optimal_gamma = self._do_grid_search(train_patterns, 
                                                            train_labels, 
                                                            self._C_grid, 
                                                            self._gamma_grid)
            #Overwrite C and gamma
            self._C = optimal_C
            self._gamma = optimal_gamma
        
        #Invoke the superclass method
        return super().predict(train_patterns, train_labels, test_patterns)
            
    
        
    def _set_model(self):
        #Set up the classification model
        self._model = sklearn.svm.SVC(C = self._C,
                                      kernel = self._kernel,
                                      gamma = self._gamma)    
            
class LinearSVMClassifier(SVMClassifier):
    """Linear Support Vector classifier"""
    
    def __init__(self, C=1.0, grid_search=False, **kwargs):      
        """The constructor
    
        Parameters
        ----------
        C : float 
            The regularization parameter. The strength of the regularization
            is inversely proportional to C. Must be strictly positive. Penalty
            is computed by L2 norm.
        grid_search : bool
            Whether to do grid search for hyperparameter optimisation. If True
            C and gamma values are overridden by the results of the grid
            search.
        C_grid : array-like (float)
            The search space for C. Required if grid_search = True, ignored
            otherwise
        """
        super().__init__(C = C, kernel = 'linear')
        self._grid_search = grid_search
        
        if self._grid_search:
            if 'C_grid' not in kwargs.keys():
                raise Exception("Search space for 'C' not given")
            self._C_grid = kwargs['C_grid']              
        
    def _do_grid_search(self, patterns, labels, C_grid):
        """Hyperparameters grid search
        
        Parameters
        ----------
        patterns :  ndarray (int, float)
            An (N,F) matrix where N is the number of patterns (observations) 
            and F the number of features
        labels : array-like (int, str)
            The pattern labels (N)
        C_grid : array-like (float)
            The search space for the penalty parameter C
            
        Returns
        -------
        optimal_C : float
            The optimal value for C
        """
        
        svc = sklearn.svm.SVC()
        clf = GridSearchCV(svc, {'C' : C_grid})
        clf.fit(patterns, labels)
        
        return clf.best_estimator_.C
    
    #Override to allow grid search if required   
    @doc_inherit
    def predict(self, train_patterns, train_labels, test_patterns):
        #Do the parameter search if required
        if self._grid_search:
            optimal_C = self._do_grid_search(train_patterns, 
                                             train_labels, 
                                             self._C_grid)
            #Overwrite C and gamma
            self._C = optimal_C
        
        #Invoke the superclass method
        return super().predict(train_patterns, train_labels, test_patterns)
             
        
    def _set_model(self):
        #Set up the classification model
        self._model = sklearn.svm.SVC(C = self._C,
                                      kernel = self._kernel)    
            
    
            
               
    