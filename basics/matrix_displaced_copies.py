import numpy as np

def matrix_displaced_copies(m, disps, filtering='circular'):
    """Displaced copies of a 2D matrix

    Parameters
    ----------
    m         : A two-dimensional R x C matrix
    disps     : An N x 2 matrix of integer representing a set of N displacement
                vectors. First column = row-wise (up-down) displacement; second
                column = column-wise (left-right) displacement
    filtering : A string indicating how the input matrix is repeated across
                borders. Can be:
                    'circular' -> circular (toroidal) repetition
                    'mirror'   -> mirrored repetition

    Returns
    -------
    layers    : An N x R x C matrix where each of the N layers represents a
                displaced copy of m as defined in disps
    """

    #Create a block matrix via a 3 x 3 repetition of m
    if filtering == 'circular':
        bm = np.bmat("m m m; m m m; m m m")
    elif filtering == 'mirror':
        tc = np.flipud(m)           #Top center block
        tl = (np.fliplr(tc)).T      #Top left block
        tr = m.T                    #Top right block
        ml = np.fliplr(m)           #Middle left block
        mc = m                      #Middle center block
        mr = ml                     #Middle right block
        bl = tr                     #Bottom left block
        bc = tc                     #Bottom center block
        br = tl                     #Bottom right block

        bm = np.bmat("tl tc tr; ml mc mr; bl bc br")
    else:
        raise Exception('Filtering not supported')

    #Origin of the layers
    rows = np.size(m,axis=0)
    cols = np.size(m,axis=1)
    x0 = rows
    y0 = cols

    #Create the layers
    nDisps = len(disps[:,0])
    layers = np.zeros((nDisps,rows,cols))
    for d in range(nDisps):
        xStart = x0 + disps[d,0];
        yStart = y0 + disps[d,1];
        xEnd = xStart + rows;
        yEnd = yStart + cols;
        layers[d,:,:] = bm[xStart:xEnd,yStart:yEnd]

    #Roll the axes (layer-last indexing)  
    layers = np.moveaxis(layers, 0, 2)
    return layers