import numpy as np
import matplotlib.pyplot as plt
import cenotaph.colour_preprocessing.colour_augmentation as ca
from cenotaph.basics.base_classes import Image

img_file = '../images/peppers.jpg'
img_in = Image(img_file)

orig_fig = plt.figure()
plt.imshow(img_in.get_data())
plt.title('Original image')

#Define a grid of displacements
f = 0.60
disps = np.array([0.0, f])
I, II, III = np.meshgrid(disps, disps, disps)
I, II, III = np.reshape(I, (I.size,1)), np.reshape(II, (II.size,1)), \
             np.reshape(III, (III.size,1))
grid = np.concatenate((I,II,III),axis=1)

perturbation_type = 'translational'

for disp in grid:
    #perturbation = ca.TranslationalPerturbation(tuple(disp[:]))
    pca_colour_augmenter = ca.PCAColourAugmenter(perturbation_type,\
                                                 disp_values = disp[:])
    perturbed_img = pca_colour_augmenter.get_result(img_in)
    fig = plt.figure()
    fig.patch.set_visible(False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')   
    plt.imshow(perturbed_img[0].get_data())
    plt.title('Perturbed image; perturbation = ' + str(disp[0]) + 'I + ' \
                                                 + str(disp[1]) + 'II + ' \
                                                 + str(disp[2]) + 'III')

plt.show()