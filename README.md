## CenOTAPH: COlour and Texture Analysis toolbox for PytHon

## Description
CenOTAPH is a collection of tools for extracting colour and texture features from still images. It includes traditional (unlearnt) visual descriptors as well as methods based on convolutional neural networks (CNN). Among the descriptors currently supported are:
- Colour descriptors:
  - Full colour histogram, marginal colour histograms, colour percentiles
- Texture descriptors based on filtering:
  - Discrete Cosine Filters, Gabor filters, Laws' masks and Zernike moments
- Texture descriptors based on Histograms of Equivalent Patterns:
  - Local Binary Patterns (LBP), Improved Local Binary Patterns (ILBP), Texture Spectrum (TS), Opponent-colour Local Binary Patterns (OCLBP) and Improved Opponent-colour Local Binary Patterns (IOCLBP).
- Image features from intermediate representations of CNN (based on [Keras](https://keras.io/)):
  - DenseNet121, MobileNet, ResNet50, VGG16 and Xception.

CenOTAPH also contains functions and classes for:
- Image preprocessing such as flip, crop, split and rotation
- One-class and multi-class classification;
- Accuracy estimation of classifiers
- Statistics and combinatorics

## Usage
The `master` branch stores the official release history. Please consider that the toolbox is not thoroughly tested and can therefore contain serious errors. These may result in instability, crashes and/or data loss (please see also the [Disclaimer](disclaimer) below).

## Dependencies
_Under construction_

## License
The code in this repository is distributed under [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)

## <a name="disclaimer">Disclaimer</a>
The information and content available on this repository are provided with no warranty whatsoever. Any use for scientific or any other purpose is conducted at your own risk and under your own responsibility. The authors are not liable for any damages - including any consequential damages - of any kind that may result from the use of the materials or information available on this repository or of any of the products or services hereon described.
