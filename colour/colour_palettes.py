"""Methods for extracting colour palettes"""
from abc import ABC, abstractmethod
from itertools import product

from sklearn.cluster import KMeans as SKLearnKMeans

from cenotaph.basics.base_classes import Image, ImageType

import numpy as np

class ColourPalette(ABC):
    """Abstract base class for computing colour palettes"""
    
    @abstractmethod
    def _compute_palette(self, img):
        pass

    def __init__(self, max_num_colours, frequency_cutoff):
        """
        Parameters
        ----------
        max_num_colours : int
            The maximum number of colours in the palette.
        frequency_cutoff : float
            Exclude from the palette those colours the relative frequency of
            which (as a fraction - 0,1) is below the cutoff value. Pass None
            if cutoff is not requested.
        """
        self._num_colours = max_num_colours
        self._frequency_cutoff = frequency_cutoff
        
        
    def get_palette(self, img):
        """
        Parameters
        ----------
        img : cenotaph.basics.base_classes.Image
            The input image.
            
        Returns
        -------
        palette : ndarray of float (num_colours, num_channels)
            The colour palette; each row represents one colour.
        norm_palette : ndarray of float (num_colours, num_channels)
            The normalised palette. Values are normalised in the [0,1] interval
            by dividing by (2**num_bins_per_channel - 1)
        frequency : ndarray of float (num_colours)
            The relative frequency (%) with which each colour of the palette
            appears in the input image. 
        
        Note
        ----
        The entries of palette, frequency are sorted in descending order of
        frequency - i.e.: the 1st entry is the most frequent element in the
        palette, the 2nd entry the second most frequent element in the
        palette, etc.
        """      
        #Compute the complete palette
        palette, frequency = self._compute_palette(img)
        
        #Sort in descending order of frequency
        sorted_indices = np.argsort(frequency[:,0])[::-1] 
        palette = palette[sorted_indices,:]
        frequency = frequency[sorted_indices]        
        
        #Apply the cutoff if requested
        if self._frequency_cutoff:
            selected = frequency > self._frequency_cutoff
            palette = palette[selected,:]
            frequency = frequency[selected]
        
        #Compute the normalised palette    
        norm_palette = palette/(2**img.get_bit_depth() - 1)
        
        return palette, norm_palette, frequency
    
class KMeans(ColourPalette):
    """Colour palette based on the k-means clustering in the RGB space
    
    """
    def __init__(self, num_colours, frequency_cutoff=None):
        """
        Parameters
        ----------
        num_colours, frequency_cutoff : see superclass doc
        """
        super().__init__(num_colours, frequency_cutoff)
        
    def _compute_palette(self, img):
        
        #Define the seed colours - uniformly spaced values along the main
        #diagonal of the RGB cube
        seeds = np.linspace(start = 0, stop = 2**img.get_bit_depth() - 1,
                            num = self._num_colours)
        seeds = np.tile(seeds, reps = (3,1)).T
        
        #Get the image data and reshape
        patterns = img.get_data()
        height, width = patterns.shape[0:2]
        patterns = np.reshape(patterns, 
                              newshape = (width*height, 3))
        
        #Do the clustering
        clusterer = SKLearnKMeans(n_clusters = self._num_colours, init = seeds,
                                  n_init = 1)
        labels = clusterer.fit_predict(patterns)
        
        #Get the palette
        palette = clusterer.cluster_centers_
        
        #Get the frequency
        frequency = np.zeros((self._num_colours,1))
        for i in range(self._num_colours):
            frequency[i] = np.sum(labels == i)/(height * width)
                
        return palette, frequency
        
    

class FullHistogram(ColourPalette):
    """Colour palette based on the three-dimensional colour histogram in the 
    RGB space
    
    References
    ----------
    [1] Ciocca, G., Napoletano, P., Schettini, R.
        (2019) Lecture Notes in Computer Science (including subseries Lecture 
        Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 
        11418 LNCS, pp. 165-179.
    """
    
    def __init__(self, num_colours, num_bins_per_channel, frequency_cutoff):
        """
        Parameters
        ----------
        num_colours, frequency_cutoff : see superclass doc
        num_bins_per_channel : int, array-like (num_channels)
            An iterable indicating the number of bins for each channel.
        """
        super().__init__(num_colours, frequency_cutoff)
        self._num_bins_per_channel = num_bins_per_channel
        
    def _compute_palette(self, img):
        
        n_channels = img.get_n_channels()
        raw_data = img.get_data(copy = False)
        palette = np.empty((n_channels, 0))
        
        #Define the bin edges and centroids
        bin_edges = list()
        bin_centroids = list()
        for ch in range(n_channels):
                        
            bin_edges_ = np.linspace(start = 0, 
                                     stop = 2**img.get_bit_depth() - 1,
                                     num = self._num_bins_per_channel[ch] + 1)
            bin_centroids_ = (bin_edges_[0:-1] + bin_edges_[1::])/2
            bin_edges.append(bin_edges_)
            bin_centroids.append(bin_centroids_)
        
        #Reshape the data
        raw_data = np.reshape(a = raw_data, 
                              newshape = (raw_data.shape[0]*raw_data.shape[1], 
                                          n_channels))
        
        #Compute the histogram
        frequency, _ = np.histogramdd(sample = raw_data, 
                                      bins = bin_edges)
        frequency = frequency.flatten()/raw_data.size
        
        #Generate the full palette
        palette = np.array(list(product(*bin_centroids)))
        
        #Sort by frequency in descending order
        sorted_indices = np.argsort(a = frequency)
        sorted_indices = np.flip(sorted_indices)
        sorted_indices = sorted_indices[0:self._num_colours]
        
        return palette[sorted_indices,:], frequency[sorted_indices]
