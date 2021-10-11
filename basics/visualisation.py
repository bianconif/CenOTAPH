import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cenotaph.basics.generic_functions as gf

def rgb_scatter_plots(img, bit_depth):

    """Scatter plots of RGB pixel values projected on the RG, RB and GB planes
        
    Parameters
    ----------
    img : ndarray, int
        The input RGB image. An H x W x 3 matrix
    bit_depth : int
        The number of quantisation levels (bit depth) per channel
    """ 
    
    G = 2**bit_depth
    
    #Unfold the input image
    unfolded_img = gf.image_reshape(img)
    r, g, b = unfolded_img[:,0], unfolded_img[:,1], unfolded_img[:,2]
    
    #Get the centroid
    centroid = [np.mean(r,axis=0), np.mean(g,axis=0), np.mean(b,axis=0)]
    
    #Compute the eigenvalues and eigenvectors via the inertia matrix
    _, eigvals, eigvecs = gf.inertia_matrix(unfolded_img, np.tile(1/r.size, r.size))
    
    #Compute the corresponding radii of giration (overall mass is one - see
    #previous line)
    radii_of_giration = np.sqrt(eigvals)
    eigvals = np.tile(eigvals, (eigvecs.shape[0],1))
    princomps = np.multiply(np.tile(radii_of_giration, (eigvecs.shape[0],1)), eigvecs)
    
    #Plot the input image
    orig_fig = plt.figure()
    plt.imshow(img)
    plt.title('Original image')
    
    #Three dimensional scatter plot (colour histogram)
    three_dim_plot = plt.figure()
    ax = three_dim_plot.add_subplot(111, projection='3d')
    plt.title('Colour distribution in the RGB cube')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.scatter(r, g, b, s = 5, facecolors = unfolded_img/(G-1))  
    
    ax.plot([centroid[0], centroid[0] + princomps[0,0]], \
            [centroid[1], centroid[1] + princomps[1,0]], \
            zs = [centroid[2], centroid[2] + princomps[2,0]], \
            color='cyan', linewidth = 2)   
    
    ax.plot([centroid[0], centroid[0] + princomps[0,1]], \
            [centroid[1], centroid[1] + princomps[1,1]], \
            zs = [centroid[2], centroid[2] + princomps[2,1]], \
            color='magenta', linewidth = 2)  
    
    ax.plot([centroid[0], centroid[0] + princomps[0,2]], \
            [centroid[1], centroid[1] + princomps[1,2]], \
            zs = [centroid[2], centroid[2] + princomps[2,2]], \
            color='yellow', linewidth = 2)      
    
    #Scatter plot on RG plane    
    rg_plot = plt.subplots()
    plt.title('Colour distribution -- RG')
    plt.scatter(r, g, s = 5, facecolors = unfolded_img/(G-1))
    plt.xlabel('Red')
    plt.ylabel('Green')
    plt.arrow(centroid[0],centroid[1],princomps[0,0],princomps[1,0])     #1st principal component
    plt.arrow(centroid[0],centroid[1],princomps[0,1],princomps[1,1])     #2nd principal component
    plt.arrow(centroid[0],centroid[1],princomps[0,2],princomps[1,2])     #3rd principal component
    
    #Scatter plot on RB plane    
    rb_plot = plt.subplots()
    plt.title('Colour distribution -- RB')
    plt.scatter(r, b, s = 5, facecolors = unfolded_img/(G-1))
    plt.xlabel('Red')
    plt.ylabel('Blue')
    plt.arrow(centroid[0],centroid[2],princomps[0,0],princomps[2,0])     #1st principal component
    plt.arrow(centroid[0],centroid[2],princomps[0,1],princomps[2,1])     #2nd principal component
    plt.arrow(centroid[0],centroid[2],princomps[0,2],princomps[2,2])     #3rd principal component
    plt.show()    
    
    #x, y = np.random.random((2, 10))
    #rgb = np.random.random((10, 3))
    
    #fig, ax = plt.subplots()
    #ax.scatter(x, y, s=200, facecolors=rgb)
    #plt.show()    
    
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()