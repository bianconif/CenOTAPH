import matplotlib.pyplot as plt
import numpy as np

from cenotaph.basics.digital_circle import andres_digital_circle
#from cenotaph.basics.digital_disc import digital_disc
from cenotaph.basics.generic_functions import t_wrap

class Neighbourhood:
    """Neighbourhood of points in two dimensions"""
    
    def __init__(self, points):
        """Constructor based on a predefined list of points
        
        Parameters
        ----------
            points : ndarray
                An N x 2 array of numbers (int or float) where each row represents 
                the coordinates of one point
        """
        
        self.set_points(points)
            
    def delete_points(self, indices):
        """Delete points based on their indices
        
        Parameters
        ----------
        indices : array_like (int)
            The indices of the points to be removed
        """
        
        self._points = np.delete(self._points, indices, 0)
        
    def get_points(self):
        """
        Points coordinates
        
        Returns
        ----------
        points : ndarray
            An N x 2 ndarray where each row represents the coordinates of one 
            point
        """    
        
        points = np.copy(self._points)
        return points
    
    def set_points(self, points):
        """Set the points coordinates
        
        Parameters
        ----------
            points : ndarray
                An N x 2 array of numbers (int or float) where each row represents 
                the coordinates of one point
        """
        
        self._points = np.copy(points)
    
    def get_integer_points(self):
        """
        Rounded (integer) points coordinates
        
        Returns
        ----------
        int_points : ndarray of int
            An N x 2 ndarray of int where each row represents the
            coordinates of one point in the neighbourhood.
        """
        
        int_points = np.rint(self._points)
        
        return int_points.astype(int)
        
    def translate(self, angle, displacement):
        """
        Translate along a line
        
        Parameters
        ----------
        angle : float
            The angle indicating the direction along which to translate. Zero 
            indicates a translation along the horizontal axis.
        displacement : float
            The entity of the displacement.          
        """
        
        #Create a copy of the points
        cached_points = self._points
        
        #Rotate by -angle degrees
        self.rotate(-angle)
        
        #Translate along the horizontal axis
        self._points = self._points + np.array([displacement, 0])
        
        #Rotate back by angle
        self.rotate(angle)
        
        #Store the fixed points
        self._fixed_points = self.compute_fixed_points(cached_points, self._points)
        

    def rotate(self, angle):
        """
        Rotate around the origin by the given angle
        
        Parameters
        ----------
        angle : float
            The angle of rotation in degrees. Positive values indicate a 
            counter-clockwise rotation.
        """
        
        #Create a copy of the points
        cached_points = self._points        
        
        #Create the transformation matrix
        M = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                      [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])
        
        #Apply the tranformation
        self._points = np.einsum('ji,kj', self._points.T, M)
        
    def reflect(self, A, B, C):
        """Reflect about the line of equation Ax + By + C = 0"""
        
        #Create a copy of the points
        cached_points = self._points          
        
        #Create the transformation matrix
        M = np.array([[B**2 - A**2, -2*A*B], [-2*A*B, A**2 - B**2]])
        M = np.divide(M, A**2+B**2)
        
        #Apply the tranformation
        self._points = np.einsum('ji,kj', self._points.T, M) + np.array([-2*A*C,-2*B*C]) 
         
    def show(self):
        """Display the neighbourhood"""
        
        plt.plot(self._points[:,0],self._points[:,1], marker='.', color='k', linestyle='none')
        #for r in range(len(self._points[:,0])):
            #if r in self._fixed_points:
                ##Fixed points
                #plt.plot(self._points[r,0],self._points[r,1], marker='p', color='r', linestyle='none')
            #else:
                ##Other points
                #plt.plot(self._points[r,0],self._points[r,1], marker='.', color='k', linestyle='none')
                
            #plt.text(self._points[r,0], self._points[r,1], str(r), fontsize=12)
        
                ##plt.plot(self._points[:,0],self._points[:,1], marker='.', color='k', linestyle='none')
        #for r in range(len(self._points[:,0])):
            #if r in self._fixed_points:
                ##Fixed points
                #plt.plot(self._points[r,0],self._points[r,1], marker='p', color='r', linestyle='none')
            #else:
                ##Other points
                #plt.plot(self._points[r,0],self._points[r,1], marker='.', color='k', linestyle='none')
                
            #plt.text(self._points[r,0], self._points[r,1], str(r), fontsize=12)
        
        plt.title('Neighbourhood')
        plt.show()    
    
    #@staticmethod  
    #def compute_fixed_points(original_points, transformed_points):
        #"""Indices of the points fixed by the transform
        
        #Parameters
        #----------
        #original_points : ndarray (N x 2) 
            #The coordinates of the points before the transform.
        #transformed_points : ndarray (N x 2) 
            #The coordinates of the points after the transform.

        #Returns
        #-------
        #indices : ndarray 
            #The indices of the points fixed by the transform.
        #"""
        
        #same_x = original_points[:,0] == transformed_points[:,0]
        #same_y = original_points[:,1] == transformed_points[:,1]
        #same_coords = np.logical_and(same_x, same_y)
        #indices = np.where(np.logical_and(same_x,same_y))[0]
        
        #return indices
        
    @staticmethod
    def compare(neighbourhood_1, neighbourhood_2):
        """Point-by-point comparison of two neighbourhoods
        
        Parameters
        ----------
            neighbourhood_1 : Neighbourhood
                The first neighbourhood to compare
            neighbourhood_2 : Neighbourhood
                The second neighbourhood to compare
                
        Returns
        -------
            indices : ndarray
                The indices of the points that are equal
        """
        
        points_1 = neighbourhood_1.get_points()
        points_2 = neighbourhood_2.get_points()
        
        if not (points_1.shape == points_2.shape):
            raise Exception('The two neighbourhoods must have the same number \
            of points') 
        
        same_x = points_1[:,0] == points_2[:,0]
        same_y = points_1[:,1] == points_2[:,1]
        same_coords = np.logical_and(same_x, same_y)
        indices = np.where(np.logical_and(same_x,same_y))[0] 
        
        return indices
    
class DigitalCircleNeighbourhood(Neighbourhood):
    """Digital circle neighbourhood based on Andres' formulation"""   
    
    def __init__(self, radius=1, max_num_points=None, full=False):
        """
        Default constructor
        
        Parameters
        ----------
        radius : int
            The radius (in pixels) of the neighbourhood
        max_num_points : int
            The maximum number of points allowed. If None the number of points
            is what results from radius and norm_exp. 
        full : bool
            Whether to include the center (True) or not (False)
   
        Effects
        -------
        Generates an N x 2 ndarray of int where each row represents the
        coordinates of one point in the neighbourhood. 
        The points are stored in the _points variable
        """
        
        self._full = full
        self._radius = radius
        
        self._center_index = None
        if full:
            self._num_peripheral_points = max_num_points - 1
        else:
            self._num_peripheral_points = max_num_points
        
        self._points = andres_digital_circle(self._radius,  
                                             self._num_peripheral_points)

        
        #If of full type prepend the center - i.e. point (0.0,0.0)
        if self._full:
            center = np.array([0.0,0.0])
            center = np.expand_dims(center, axis=0)
            self._points = np.concatenate((center,self._points),axis=0)
            
            #Store the index of the center point
            self._center_index = 0
            
    def is_full(self):
        return self._full
    
    def get_radius(self):
        return self._radius
        
    def get_num_points(self):
        return self._points.shape[0]
               
    def get_num_peripheral_points(self):
        n_points = self.get_num_points()
        if self._full:
            n_points = n_points - 1
        return n_points
                   
    def center_index(self):
        """Index of the central point
            
        Returns
        -------
        index : int
            The index of the central point (None if neighbourhood
            is not 'full')
        """
        return self._center_index
    
    def peripheral_indices(self):
        """Indices of the peripheral points
        
        Returns
        -------
        indices : list of int
            The indices of the peripheral points
        """
        retval = list(range(self.get_num_points()))
        if self.is_full():
            retval.remove(self.center_index())
        return retval

        
    def __repr__(self):
        retval = self.__class__.__name__\
            + '-r' + str(self.get_radius())\
            + '-p' + str(self.get_num_peripheral_points())
        if self._full:
            retval = retval + '-full'
        
        return retval
        
                                
class SquareNeighbourhood(Neighbourhood):
    """Square neighbourhood of points in two dimensions"""
    
    def __init__(self, side):
        """
        Default constructor
        
        Parameters
        ----------
        side : int
            The side length
        """
        
        radius = np.floor(side/2)
        super().__init__(radius, 'Disc-L0')
        
        #Max and min coordinate values
        self._coord_bounds = (self._points.min(), self._points.max())
        
        #Unit length
        self._unit_length = (self._coord_bounds[1] - self._coord_bounds[0])/(side - 1)
        
    def toroidal_wrap(self):
        """Apply a toroidal wrap to the points"""
        lower_bound = self._coord_bounds[0] - self._unit_length/2
        upper_bound = self._coord_bounds[1] + self._unit_length/2
        
        self._points = np.vectorize(t_wrap)(self._points, lower_bound, 
                                            upper_bound)