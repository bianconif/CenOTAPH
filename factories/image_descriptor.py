import cenotaph.colour_descriptors as cd

def image_descriptor(d_type, **kwargs):
    """Create and return an ImageDescriptor object. This function is both 
    a factory method and a facade to the image pre-processing method
    
    Parameters
    ----------
    d_type : str
        The name of the image descriptor to be generated. Accepted values:
            **********************************************************
            ****Colour descriptors - img needs to be three-channel****
            **********************************************************
            'full_hist'      -> Three-dimensional joint colour histogram
            'marg_hists'     -> Three one-dimensional colour histograms
            'mean'           -> Mean of each colour channel
            'mean_std'       -> Mean and std. dev of each colour channel
            'percentiles'    -> Percentiles of each colour channel
    **kwargs
        Optional parameters. Accepted values:
            d_type = 'full_hist', 'marg_hists'
                'nbins' : int -> Number of bins for each channel (def = 8)
            d_type = 'percentiles'
                'ncentiles' : int -> Number of percentiles for each channel 
                                     (def = 3)
            d_type = 'mean', 'mean_std'
                 No optional parameters
    Returns
    ----------
    img_descr : ImageDescriptor
        The image decsriptor
    """
    
    #Initialise the output
    img_descr = None
    
    if d_type == 'full_hist':
        if 'nbins' in kwargs.keys():
            nbins = kwargs['nbins']
            img_descr = cd.FullHist(nbins)
        else:
            img_descr = cd.FullHist()
    elif d_type == 'marg_hists':
        if 'nbins' in kwargs.keys():
            nbins = kwargs['nbins']
            img_descr = cd.MargHists(nbins)
        else:
            img_descr = cd.MargHists()
    elif d_type == 'mean':
        img_descr = cd.Mean()
    elif d_type == 'mean_std':
        img_descr = cd.MeanStd()
    else:
        raise Exception('Image descriptor not supported')
    
    return img_descr
        
