import cenotaph.symmetry_transforms as st
import matplotlib.pyplot as plt
from PIL import Image

imgfile = '../images/peppers.jpg'
img = Image.open(imgfile)
plt.imshow(img)
plt.title('Original image')
plt.show()

statistical_test = 'L2-distance-std'
resolutions = [3]
geometric_transforms = {'r90', 'r180', 'r270', 'hMirror', 'vMirror', \
                        'd1Mirror', 'd2Mirror'}

for r in resolutions:
    for gt in geometric_transforms:
        my_st = st.SymmetryTransform(imgfile, r, gt, statistical_test)
        plt.imshow(my_st.get_symmetry_map(), aspect='equal', interpolation='none')
        plt.title('Tranformed image. Res = ' + str(r) + 'px, transformation = ' + gt)
        plt.axis('off')
        plt.figure()

plt.show()
