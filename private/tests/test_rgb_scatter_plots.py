import cenotaph.basics.visualisation as vs
import numpy as np
from PIL import Image

imgfile = '../images/green-peas-redu.jpg'
img = np.asarray(Image.open(imgfile))
bit_depth = 8

vs.rgb_scatter_plots(img, bit_depth)