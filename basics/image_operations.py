"""Generic functions for image manipulation"""
from PIL import Image

def crop_to_central_square(source_file, destination_file):
    """Crops the input image to the maximal centered square
    
    Parameters
    ----------
    source_file : str
        Pointer to the source file.
    destination_file : str
        Pointer to the destination file.
    """
    
    #Read the input image
    img_in = Image.open(source_file)
    
    #Compute the side length of the cropping area
    img_size = img_in.size
    crop_length = min(img_size[0:2])
    
    #Crop the image
    img_cropped = img_in.crop(((img_size[0] - crop_length) // 2,
                               (img_size[1] - crop_length) // 2,
                               (img_size[0] + crop_length) // 2,
                               (img_size[1] + crop_length) // 2))
    
    #Save the cropped image
    img_cropped.save(destination_file)
    
    return


