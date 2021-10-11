import nupmy as np

def create_mask(n, shp):
    """Generate mask ow weights for LBP-like descriptors

    Parameters
    ----------
    n : int
        The number of pixels in the neighbourhood.
    shp : sequence of two int
        The dimension (H x W) of the input image
    
    Returns
    -------
    mask : ndarray of int
        The mask as an H x W x n matrix 
    """    

    generator = (np.arange(n)) ** 2
    mask = np.tile(generator, (shp[0],shp[1],1))