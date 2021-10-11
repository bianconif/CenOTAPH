import cenotaph.colour_preprocessing.colour_constancy as cc

def image_preprocessor(img, p_type, **kwargs):
    """Create and return an ImagePreprocessor object. This function is both 
    a factory method and a facade to the image pre-processing method
    
    Parameters
    ----------
    img : ndarray
         A single-channel (H,W) or three-channel image (H,W,3)
    p_type : str
        The name of the image pre-processor to be generated. Accepted values:
            ****Colour preprocessing - img needs to be three-channel****
            'chroma'      -> Chromaticity representation
            'grey_world'  -> Grey world colour normalisation
            'max_white'   -> Max white colour normalisation
            'stretch'     -> Colour equalisation via 'stretch' algorithm
    bit_depth : int
            The number of quantisation levels (bit depth) of the input image
            per channel
    **kwargs
        Optional parameters. Accepted values:
            p_type = 'chroma', 'grey_world', 'max_white' or 'stretch'
                No optional parameters
    Returns
    ----------
    img_preproc : ImagePreprocessor
        The image pre-processor
    """
    
    #Initialise the output
    img_preproc = None
    
    if p_type == 'chroma':
        img_preproc = cc.Chroma(img, bit_depth)
    elif p_type == 'grey_world':
        img_preproc = cc.GreyWorld(img, bit_depth)
    elif p_type == 'max_white':
        img_preproc = cc.MaxWhite(img, bit_depth)
    elif p_type == 'stretch':
        img_preproc = cc.Stretch(img, bit_depth)
    else:
        raise Exception('Image pre-processing method not supported')
    
    return img_preproc
        
        