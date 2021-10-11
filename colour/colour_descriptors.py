from cenotaph.basics.base_classes import ImageDescriptor
import numpy as np

class Percentiles(ImageDescriptor):
    """Percentiles of each colour channel

    ----------
    References
    [1] M. Niskanen, O. Silvén, and H. Kauppinen, Color and texture based
        wood inspection with non-supervised  clustering, in Proc. of
        12th Scandivanian Conf. on Image Analysis, Bergen, Norway,
        pp.336-342 (June 2001)
    """

    def __init__(self, ncentiles=3):
        """Constructor

        Parameters
        ----------
        ncentiles : int
            The number of percentiles to be computed. For instance,
            if ncentiles = 3, the 25%, 50% and 75% percentiles are
            computed; if ncentiles = 4 the 20%, 40%, 60% and 80%
            percentiles are computed.
        """
        super().__init__()
        self._ncentiles = ncentiles

        # Compute the percentages at which the percentiles are to be evaluated
        start = 0.0
        stop = 100.0
        num = self._ncentiles + 2
        percentages = np.linspace(start, stop, num)
        toRemove = [0, len(percentages) - 1]
        percentages = np.delete(percentages, toRemove)
        self._percentages = percentages

    def _compute_features(self):
        """Compute the percentiles of each colour channel"""
        
        features = np.array([])
        _n_channels = self._img_in.get_n_channels()
        _data = self._img_in.get_data()        

        if _n_channels == 1:
            features = np.percentile(_data[:, :], self._percentages)
        else:
            for ch in np.arange(_n_channels):
                features = np.append(features,
                    np.percentile(_data[:, :, ch], self._percentages)) 
                
        return features
                
    def __repr__(self):
        retval = super().__repr__() + '-{}'.format(self._ncentiles)  
        return retval    

class Mean(ImageDescriptor):
    """Mean of each colour channel

    ----------
    References
    [1] Kukkonen, S., Kälviäinen, H., Parkkinen, J.
        Color features for quality control in ceramic tile industry
        (2001) Optical Engineering, 40 (2), pp. 170-177.
    """

    def _compute_features(self):
        """Compute the mean of each colour channel"""        
        
        return np.average(self._img_in.get_data(), (0,1))

class MeanStd(Mean):
    """Mean and standard dev. of each colour channel"""

    def _compute_features(self):
        """Compute the mean and standard deviation of each colour channel"""
        
        _mean = np.average(self._img_in.get_data(), (0,1))
        _std = np.std(self._img_in.get_data(), (0,1))
        
        #Add dimension if necessary
        if _mean.ndim == 0:
            _mean = np.expand_dims(_mean, axis=0)
        if _std.ndim == 0:
            _std = np.expand_dims(_std, axis=0)            
        
        return np.concatenate((_mean, _std))



class MarginalHists(ImageDescriptor):
    """Concatenation of the marginal histograms of each colour channel

    ----------
    References
    [1] Pietikainen, M., Nieminen, S., Marszalec, E., Ojala, T.
        Accurate color discrimination with classification based on feature
        distributions (1996) Proceedings - International Conference on
        Pattern Recognition, 3, art. no. 547285, pp. 833-838.
    """

    def __init__(self, nbins):
        """Constructor

        Parameters
        ----------
        nbins : int for grey-scale images or triple of int for RGB images
            The number of bins for each marginal histogram
        """

        super().__init__()
        self._nbins = nbins

    def _compute_features(self):
        """Compute the marginal colour histograms"""
        features = np.array([])
        
        _lower_range = 0
        _upper_range = 2**self._img_in.get_bit_depth() - 1 
        
        _n_channels = self._img_in.get_n_channels()
        _n_pixels = self._img_in.get_num_pixels()
        _data = self._img_in.get_data()
        
        if _data.ndim < 3:
            _data = np.expand_dims(_data, 2)
        
        for ch in np.arange(_n_channels):
            h, _ = np.histogram(_data[:, :, ch], self._nbins[ch],
                                (_lower_range,_upper_range))
            h = np.divide(h, _n_pixels)
            features = np.concatenate((features, h), axis=0)  
            
        return features
            
    def __repr__(self):
        retval = super().__repr__()
        for bins in self._nbins:
            retval = retval + '-{}'.format(bins)   
        return retval

class FullHist(MarginalHists):
    """Full 3D colour histogram

    ----------
    References
    [1] Swain, M.J., Ballard, D.H.
        Color indexing (1991) International Journal of Computer Vision,
        7 (1), pp. 11-32.
    """

    def _compute_features(self):
        """Compute the full colour histogram"""
        
        features = np.array([])
        
        _lower_range = 0
        _upper_range = 2**self._img_in.get_bit_depth() - 1 
        
        _n_channels = self._img_in.get_n_channels()
        _n_pixels = self._img_in.get_num_pixels()
        _data = self._img_in.get_data()        

        if _n_channels == 1:
            features = super()._compute_features()
        else:
            packed_data = np.empty((_n_pixels, 3))
            range = []
            for ch in np.arange(_n_channels):
                packed_data[:,ch] = np.ndarray.flatten(_data[:, :, ch])
                range = range + [[_lower_range, _upper_range]]

            h, _ = np.histogramdd(packed_data, bins = self._nbins, range = range)
            features = np.ndarray.flatten(np.divide(h, _n_pixels))
            
        return features
            

