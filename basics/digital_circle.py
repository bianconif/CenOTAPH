from math import pi
import numpy as np
from scipy.spatial import distance_matrix
from scipy.signal import resample

from cenotaph.basics.digital_disc import digital_disc, square_lattice

def andres_digital_circle(radius=1, max_num_points=8):
    """Digital circles in two dimensions as defined in [1]

    Parameters
    ----------
    radius : int
        The (integer) radius of the circle
    max_num_points : int
        The maximum number of points in the circle (for downsampling).

    Returns
    -------
    points : ndarray of float 
         An N x 2 matrix where each row represents a point. First column = x
         coordinate; second column = y coordinate
    
    References
    ----------
    [1] Andres, E., Roussillon, T. Analytical description of digital circles
        (2011) Lecture Notes in Computer Science (including subseries Lecture 
        Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 
        6607 LNCS, pp. 235-246. 
    """    
    
    #Create the square lattice where the circle is to be embedded
    x,y = square_lattice(2*radius)
    
    #Compute the distance map from the center (0,0)    
    squared_distance = np.add(x**2,y**2)
        
    #Get the indices that define the circle
    condition_1 = (squared_distance >= (radius - 0.5)**2)
    condition_2 = (squared_distance <= (radius + 0.5)**2)
    condition = np.logical_and(condition_1, condition_2)
    circle, _ = np.where(condition)
    
    #Sort the points in counter-clockwise order
    angle = np.arctan2(y[circle], x[circle])
    sorted_indices = np.argsort(angle, axis=0)  #This add one dimension (unnecessarily)
    sorted_indices = sorted_indices[::-1]       
    sorted_indices = sorted_indices.flatten()   #Remove the extra dimension

    X = x[circle][sorted_indices]
    Y = y[circle][sorted_indices]

    points = np.hstack((X, Y))
    
    #Downsample the circle if required
    if points.shape[0] > max_num_points:
        points = resample(points, max_num_points)

    return points    
    
    
    

def digital_circle(radius, n=2, **kwargs):
    """Digital circles in two dimensions

    Parameters
    ----------
    radius : int
        The (integer) radius of the circle
    n : int
        The exponent used to compute the distance from the center. Use 0 for
        Chebyshev distance, e.g.:
            n = 0 -> 'Chebyshev' distance
            n = 1 -> 'cityblock' distance
            n = 2 -> 'Euclidean' distance
    max_num_points (optional) : int
        The maximum number of points in the circle (for downsampling).

    Returns
    -------
    An N x 2 matrix where each row represents a point. First column = x
    coordinate; second column = y coordinate
    """

    # Create the outer and inner discs
    _,_,x,y,outer_disc = digital_disc(radius, 2*radius, n)
    _,_,_,_,inner_disc = digital_disc(radius - 1, 2*radius, n)

    #Remove the inner disc from the outer one
    not_inner_disc = np.logical_not(inner_disc)
    circle = np.logical_and(outer_disc, not_inner_disc)

    #Sort the points in counter-clockwise order
    angle = np.arctan2(y[circle], x[circle])
    sorted_indices = np.argsort(angle, axis=0)  #This add one dimension (unnecessarily)
    sorted_indices = sorted_indices[::-1]       
    sorted_indices = sorted_indices.flatten()   #Remove the extra dimension

    X = x[circle][sorted_indices]
    Y = y[circle][sorted_indices]

    M = np.hstack((X, Y))
    
    #Downsample the circle if required
    try:
        max_num_points = kwargs['max_num_points']
        M = downsample_circle(M, max_num_points)
    except:
        #Do nothing
        pass

    return M

def downsample_circle(circle, num_points, alpha=0.0):
    """Downsaple digital circle
    
    Parameters
    ----------
    circle : ndarray (N,2)
        The coordinates of the points of the circle to downsample
    num_points : int
        The number of points of the output circle
    alpha : float
        The offset angle from which the removal starts
    
    Returns
    -------
    output_circle : ndarray (num_points, 2)
        The coordinates of the points of the downsampled circle
    """
    
    output_circle = np.copy(circle)
    
    num_original_points = circle.shape[0]
    num_points_to_remove = num_original_points - num_points
    if num_points_to_remove < 0:
        raise Exception('The number of points to remove needs fewer'
                        'than there are in the original circle')
    elif num_points_to_remove == 0:
        #Do nothing
        pass
    else:
        #Get the original angles
        angles = np.arctan2(circle[:,1], circle[:,0])
        
        #Sort angles and corresponding coordinates in ascending order
        sorted_idx = np.argsort(angles)
        angles = angles[sorted_idx]
        circle = circle[sorted_idx,:]
                
        #Define the angular span between the points to be removed
        span = 2*pi/(num_points_to_remove)
        
        #Compute the angle of the points to be removed, starting from
        #the first point as the origin
        angles_to_remove = np.asarray([(angles[0] + alpha + x*span) 
                                      for x in np.arange(num_points_to_remove)])
        
        #Compute the distance between the original angles and those
        #to remove
        dm = distance_matrix(np.expand_dims(angles, axis=-1), 
                             np.expand_dims(angles_to_remove, axis=-1), 
                             p=1)
  
        #Get the minimum distance to identify the candidate points for
        #removal
        min_dist = np.min(dm, axis=1)
        
        #Sort in ascending order and get the indices of the points to remove
        to_remove = np.argsort(min_dist)[0:num_points_to_remove]
        
        #Remove the points
        output_circle = np.delete(output_circle, to_remove, axis=0)
        
    return output_circle 
        
       
