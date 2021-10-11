import matplotlib.pyplot as plt
import cenotaph.colour_descriptors as cd
from cenotaph.basics.base_classes import Image

imgfile = '../images/peppers.jpg'
#imgfile = '../images/landscape-GS-8.jpg'
img_in = Image(imgfile)

testMean = cd.Mean()
F = testMean.get_features(img_in)
print('Mean features of ' + imgfile + ' = ' + str(F))

testMeanStd = cd.MeanStd()
F = testMeanStd.get_features(img_in)
print('MeanStd features of ' + imgfile + ' = ' + str(F))

nbins = (3,3,3)
testMarginalHists = cd.MarginalHists(nbins)
F = testMarginalHists.get_features(img_in)
print('MarginalHists features of ' + imgfile + ' = ' + str(F))

nbins = (3,3,3)
testFullHist = cd.FullHist(nbins)
F = testFullHist.get_features(img_in)
print('FullHist features of ' + imgfile + ' = ' + str(F))

ncentiles = 3
testPercentiles = cd.Percentiles(ncentiles)
F = testPercentiles.get_features(img_in)
print('Percentiles of ' + imgfile + ' = ' + str(F))

##Test conversion to greyscale
#gs = rgb2gray(testPercentiles.img)
##cplot = plt.imshow(testPercentiles.img)
#plt.show()
#gsplot = plt.imshow(gs, cmap='gray')
#plt.show()
#a = 0
#a = 2