from itertools import permutations
from math import ceil, log
import glob
import os

import numpy as np
from scipy.stats import mannwhitneyu
from scipy.spatial import distance_matrix
from sklearn.metrics import confusion_matrix, jaccard_score

def similarity_measures_bw(estimated, target):
    """Similarity measures to compare binary images.
    
    Parameters
    ----------
    estimated : np.array (H,W) of uint in the [0,1] range
        The estimated values; usually the result of a segmentation procedure. 
    terget : np.array (H,W) of uint in the [0,1] range
        The target values (ground truth). 
    The convention is: 0 = 'negative'; 1 = 'positive'.
        
    Returns
    -------
    dice : float [0,1]
        Sorensen-Dice coefficient [1]
    jaccard : float [0,1]
        Jaccard index [1]
        
    References
    ----------
    [1] Verma, V., Aggarwal, R.K. A comparative analysis of similarity 
        measures akin to the Jaccard index in collaborative recommendations: 
        empirical and theoretical perspective (2020) Social Network Analysis
        and Mining, 10 (1), art. no. 43. 
    """
    
    #Make sure the result of the segmentation and the ground truth have the 
    #same shape
    if not (estimated.shape == target.shape):
        raise Exception('The estimated and target matrices must have the same'
                        'size')
    
    #Make sure the input matrices are two-dimensional
    if not (estimated.ndim == 2):
        raise Exception('The input matrices must be two-dimensional')
    
    #Flatten the input
    estimated = estimated.flatten()
    target = target.flatten()
    
    #Compute the confusion matrix
    cm = confusion_matrix(target, estimated, labels = [0,1], normalize = 'all')
    cm = np.array(cm).flatten()
    
    #Get true positives, false positives, etc. from the confusion matrix
    TN, FP, FN, TP = cm
    
    #Compute the measures
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    jaccard = jaccard_score(y_true = target, y_pred = estimated)
    dice = 2*jaccard/(1 + jaccard)  #[1, Eq. 10]
    
    return dice, jaccard, sensitivity, specificity

def multilevel_thresholding(data_in, thresholds, equalities=False):
    """Multilevel threshoding on the input data given a set of strictly 
    increasing thresholds T = [t0,...,tn]. Returns the 
    
    Parameters
    ----------
    data_in : ndarray of int or float
        The data to be thresholded.
    thresholds : one-dimensional ndarray of int or float
        The threshold values.
    equalities : bool
        Whether to consider specific levels when the input data are exactly
        equal to the threshold values.
    
    Returns
    -------
    levels : ndarray of int (same size as data_in)
        The results of the thresholding. For each X in data_in the output 
        levels will be:
            0      if -Inf    < X <= t0
            1      if t0      < X <= t1
            ...
            i      if t_{i-1} < X <= t_{i}
            ...
            n      if tn      < X <= Inf
        *** if equalities == false ***
        and
            0      if -Inf    < X < t0
            1      if           X = t0
            2      if t0      < X < t1
            ...
            2i - 1 if           X = t_{i}
            2i     if t_{i}   < X < t_{i+1} 
            ...
            2n     if tn      < X <= Inf
        *** if equalities == true ***   
        The total number of different levels is therefore (n + 1) if
        equalities == False and (2n + 1) i equualities == True.
    """
    
    #Make a copy
    thresholds = thresholds[:]
    
    #Sort the thresholds in ascending order
    thresholds.sort()
    
    #Add infinite boundaries
    thresholds.append(np.Inf)
    thresholds.insert(0, -np.Inf)
        
    #Threshold the differences
    levels = np.zeros(data_in.shape, dtype='int')
    for i in range(len(thresholds) - 1):
        if equalities:
            if (i != 0):
                indices_equal = np.where(data_in == thresholds[i])
                levels[indices_equal] = 2*i - 1
            
            indices_within = np.where(np.logical_and((data_in > thresholds[i]),
                                                     (data_in < thresholds[i+1])))  
            levels[indices_within] = 2*i
        else:
            indices = np.where(np.logical_and((data_in >= thresholds[i]),
                                              (data_in < thresholds[i+1])))
            levels[indices] = i
        
    return levels

def convert_base(data_in, new_base):
    """Convert an array of decimal numbers to a different base
    
    Parameters
    ----------
    data_in : ndarray of int (N)
        The input data in decimal base
    new_base : int
        The new base
        
    Returns
    -------
    data_out : ndarray of int (N,M)
        The data the new base. Each row is the representation of the
        N-th input number in the new base. M is determined by the maximum
        input value. For instance: if max(data_in) = 256 and new_base = 2 then
        M = 8; if max(data_in) = 81 and new_base = 3 then M = 4.
    """
    
    #Compute the number of digits needed to represent the input data
    max_value = np.max(data_in)
    n_digits = ceil(log(max_value, new_base))
    
    #Create the weight (power) mask in the new base
    weights = new_base ** np.arange(n_digits)
    weights = weights[::-1] #Reverse the order
    weights = np.tile(weights, (len(data_in), 1))
    
    #Results of the integer division between the input data and the weights
    div_results = np.zeros((len(data_in), n_digits), dtype = int)
    div_results[:,0] = data_in // weights[:,0]
        
    #Remainders of the divisions between the input data and the powers
    #of new_base
    remainders = np.zeros((len(data_in), n_digits), dtype = int)    
    remainders[:,0] = data_in % weights[:,0]
    
    for i in range(1, n_digits):
        div_results[:,i] = remainders[:,i-1] // weights[:,i]
        remainders[:,i] = remainders[:,i-1] % weights[:,i]
    
    return div_results

def cast_image_type(img_in, bit_depth, copy=True):
    """Casts the inner image type according to its bit depth
    
    Parameters
    ----------
        img_in : ndarray
            The input image
        bit_depth : int
            The number of bits used to encode each colour channel
        copy : boolean
            Whether to return a copy (newly allocated array) or not 
            
    Returns
    -------
        img_out : ndarray
            The output image, same size as the input one
    """
    
    _accepted_bit_depths = {8:'uint8', 16:'uint16', 32:'uint32'}
    
    if not bit_depth in _accepted_bit_depths.keys():
        raise Exception('Bit depth not supported')
    
    img_out = img_in.astype(_accepted_bit_depths[bit_depth], copy)
    
    return img_out
        
def image_reshape(img_in):
    """Image unfolding. Reshapes the H x W x c input image into a (H x W) x c
    matrix
    
    Parameters
    ----------
        img_in : ndarray
            The input image
            
    Returns
    -------
        img_out : ndarray
            The reshaped image
    """    
    
    shp = img_in.shape
    num_pixels = shp[0]*shp[1]
    num_channels = shp[2] 
    img_out = img_in.reshape((num_pixels,num_channels))
    
    return img_out


def t_wrap(x, lower_bound, upper_bound):
    """Toroidal wrap
    
    Parameters
    ----------
        x : int or float
            The value to be wrapped
        lower_bound : int or float
            The lower bound of the interval where x is to be mapped
        upper_bound : int of float
            The upper bound of the interval where x is to be mapped
            
    Returns
    -------
        xw : int or float
            The value of x mapped into the [lower_bound, upper_bound] interval
    """
    
    span = upper_bound - lower_bound
    
    if x > upper_bound:
        xw = lower_bound + ((x - upper_bound) % span)
    elif x < lower_bound:
        xw = upper_bound - ((lower_bound - x) % span)
    else:
        xw = x
        
    return xw


def combine_patterns(pattern_map_1, pattern_map_2, operation, **kwargs):
    """Element-wise operation on two pattern maps
    
    Parameters
    ----------
    pattern_map_1 : ndarray
        The first operand. An H x W x N matrix representing a set of 
        N-dimensional patterns over a discrete domain of dimension H x W.
    pattern_map_2 : ndarray
        The second operand. Must be the same type and size of pattern_map_1.
    operation : str
        The type of element-wise operation to perform between the two operands.
        Can be:
            'distance' -> distance in the N-dimensional space. Use **kwargs to 
            specify the type of distance.
    **kwargs
        List of additional keyworded arguments whose possible values depend on 
        the operation requested.
            If operation == 'distance'
                key = 'n'
                value = {non-zero int, inf, -inf} (the order of the norm)
    
    Returns
    -------
    output_map : ndarray
        An H x W matrix representing the element-wise results of pattern_map_1
        operated pattern_map_2
    """
    
    #Check the two input maps are the same size
    if not (pattern_map_1.size == pattern_map_2.size):
        raise Exception('The two pattern maps must be the same size')
    
    dims = pattern_map_1.shape
    output_map = np.empty(dims) 
    
    if operation == 'distance':
        if 'n' not in kwargs:
            raise Exception('Order of the norm not specified')
        else:
            n = kwargs['n']
            output_map = np.linalg.norm(pattern_map_1 - pattern_map_2, n, 0)
    else:
        raise Exception('Operation type not supported')
    
    return output_map


def pca(points):
    """Principal Components Analysis
    
    Parameters
    ----------
    points : ndarray
        The coordinates of the points. An N x 3 matrix where rows represent points
        and columns coordinates.
    
    Returns
    -------
    eigvals : ndarray
        The eigenvalues of the inertia matrix sorted in descending order.
    eigvecs : ndarray
        The eigenvectors packed in a 3 x 3 matrix arranged in column-wise order. 
        Each column represents one eigenvector: the first column the eigenvector
        with the highest eigenvalue, the second column the eigenvector with the 
        second-highest eigenvalue, etc. 
    """     

    n, m = points.shape
    
    #Make sure the input matrix is zero-mean
    if not np.allclose(points.mean(axis=0), np.zeros(m)):
        centroids = points.mean(axis=0)
        points = points - centroids
        
    #Compute covariance matrix
    C = np.dot(points.T, points) / (n-1)

    #Eigen decomposition
    eigvals, eigvecs = np.linalg.eig(C)
    
    #Sort the eigevalues and vectors in descending order of strength
    sorted_idxs = np.argsort(eigvals)
    eigvals = eigvals[sorted_idxs[::-1]]
    eigvecs = eigvecs[:,::-1]
    
    return eigvals, eigvecs    

    
def inertia_matrix(points, mass_distr, central=True):
    """Inertia matrix of a three-dimensional distribution of mass points
    
    Parameters
    ----------
    points : ndarray
        The coordinates of the points. An N x 3 matrix where rows represent points
        and columns coordinates.
    mass_distr : ndarray
        The mass distribution. An N x 1 matrix where each value represents the mass
        of the corresponding point.
    central : boolean
        Whether we want the inertia matrix central (origin located at the center
        of mass of the point distribution) or not.
    
    Returns
    -------
    in_matr : ndarray
        The inertia matrix. A 3 x 3 matrix.
    eigvals : ndarray
        The eigenvalues of the inertia matrix sorted in descending order.
    eigvecs : ndarray
        The eigenvectors packeed in a 3 x 3 matrix arranged in column-wise order. 
        Each column represents one eigenvector: the first column the eigenvector
        with the highest eigenvalue, the second column the eigenvector with the 
        second-highest eigenvalue, etc. 
    """    
    
    if not (points.shape[0] == mass_distr.shape[0]):
        raise Exception('Points and mass distribution do not match')
    
    if central:
        #Compute the centroid first
        total_mass = mass_distr.sum()
        centroid = np.array([np.dot(points[:,0], mass_distr).sum(), \
                             np.dot(points[:,1], mass_distr).sum(), \
                             np.dot(points[:,2], mass_distr).sum()])/total_mass
        
        #Translate the origin to the centroid
        centroid = np.tile(centroid,(points.shape[0],1))
        points = points - centroid
        
    #Compute the inertia matrix
    Ix = np.multiply(np.add(np.power(points[:,1],2), np.power(points[:,2],2)),mass_distr).sum()
    Iy = np.multiply(np.add(np.power(points[:,0],2), np.power(points[:,2],2)),mass_distr).sum()
    Iz = np.multiply(np.add(np.power(points[:,0],2), np.power(points[:,1],2)),mass_distr).sum()
    Ixy = - np.multiply(np.multiply(points[:,0],points[:,1]),mass_distr).sum()
    Ixz = - np.multiply(np.multiply(points[:,0],points[:,2]),mass_distr).sum()
    Iyz = - np.multiply(np.multiply(points[:,1],points[:,2]),mass_distr).sum()
    in_matr = np.array([[Ix, Ixy, Ixz], [Ixy, Iy, Iyz], [Ixz, Iyz, Iz]])
    
    #Compute the eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(in_matr)
    
    #Sort eigenvalues by strength in descending order (highest first)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:,::-1].copy()
    
    return in_matr, eigvals, eigvecs
            
def is_file_in_folder(filename, target_folder):
    """Check if a file is in the given folder or in any of the subfolders
    
    Parameters
    ----------
    filename : str
        The name of the file to be searched (extension included)
    target_folder : str
        The absolute or relative path of the folder where to look for the file
        
    Return
    ------
    found : bool
        True if the file has been found, False otherwise
    """
    
    found = False
    
    for root, dirs, files in os.walk(target_folder):  
        if filename in files:
            found = True
            break
        
    return found

def clear_folder(folder):
    """Clear the content of a folder
    
    Parameters
    ----------
    folder : str
        Path to the folder to clear
    """
    
    files = glob.glob(folder)
    for f in files:
        os.remove(f)    

def get_folders(root):
    """Return the folders at the zero level of the given path
    
    Parameters
    ----------
    root : str
        Path
        
    Returns
    -------
    folders : list (str)
        List of folder names
    """    
    return [d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]

def get_files_and_folders(root):
    """Return the files' and folders' names in a directory tree
    
    Parameters
    ----------
    root : str
        Path to the root of the directory tree
        
    Returns
    -------
    files : list (str)
        The full paths to the files in the directory tree
    folders : list (str)
        The names of the first-level folder containing the corresponding files
    """
    
    #Get the full path to the files first
    files = get_files_in_folder(root)
    
    #Strip the input folder from the path and retain what remains until the
    #first '/'
    folders = list()
    for file in files:
        tail = file[(len(root) + 1):]
        head = tail[:tail.find('/')]
        folders.append(head)
          
    return files, folders
    
    
def get_files_in_folder(folder):
    """List of the full paths to the files contained in the given folder
        
    Parameters
    ----------
    folder : str
        The input folder
            
    Returns
    -------
    files : list (str)
        The list of the files (full paths) contained in the input folder. The 
        paths are relative to the input folder and include this one
    """
    
    files = list()
        
    for dirName, subdirList, fileList in os.walk(folder):
        for fname in fileList:
            files.append((dirName + '/' + fname).replace('\\','/'))
             
    return files

def mann_whitney_two_sided(x, y, alpha):
    """Two-sided Mann-Whitney U Test (a.k.a. Wilcoxon Rank Sum Test)
    
    Parameters
    ----------
    x, y : list of float
        Array of samples representing the two population. Should be 
        one-dimensional.
    alpha : float
        The significance threshold
        
    Returns
    -------
    p_value : float
        The p-value
    significant : bool
        Whether the difference is significant or not
    """

    #Define the 'zero hypothesis'
    x_greater_than_y = False
    y_greater_than_x = False

    #One-tailed test x > y
    _, p_value = mannwhitneyu(x, y, alternative = 'greater')
    if p_value < alpha:
        x_greater_than_y = True
    else:
        #One-tailed test y > x
        _, p_value = mannwhitneyu(y, x, alternative = 'greater')
        if p_value < alpha:
            y_greater_than_x = True

    #Set significance flag    
    significant = False
    if x_greater_than_y or y_greater_than_x:
        significant = True    
        
    return p_value, significant

def pairwise_distances(matrix, p=2, verbose=False):
    """Pairwise distances (distance matrix)
    
    Parameters
    ----------
    matrix : np.array (N,F)
        A matrix of N vectors (observations) in an F-dimensional space.
    p : int
        The exponent of the Minkowsky distance.
    verbose : bool
        If True messages are printed to track progression.
        
    Returns
    -------
    distances : list of float 
        A list of length binom(N,2) containing the pairwise distances.
    pairs : list of tuples
        A list of tuples the same length as 'distances'. Each i-th tuple 
        contains the row indices of the two vectors for which the mutual
        distance is stored in the i-th entry of 'distances'.
    """
    
    N, F = matrix.shape
    distances = list()
    
    #Get the 2-combinations row-by-row
    pairs = list(permutations(np.arange(0, N), 2))
    
    #Compute the pairwise distances
    num_pairs = len(pairs)
    
    #If verbose set up a waitbar
    if verbose:
        diff_counter = 0.0
        print('Computing pairwise distances: ', end = '')    
    
    for i, pair in enumerate(pairs):

        x = np.reshape(matrix[pair[0], :], (1, F))
        y = np.reshape(matrix[pair[1], :], (1, F))        
        distance = distance_matrix(x, y, p)[0][0]
        distances.append(distance)
        
        #Waitbar: write an asterisk each time 5% is done
        if verbose:
            diff_counter = diff_counter + 1/num_pairs
            if diff_counter >= 0.05:
                print('*', end = '')
                diff_counter = 0.0
    if verbose:
        print('\nDone!')
    
    return distances, pairs
   
    
    
  
    
    

        