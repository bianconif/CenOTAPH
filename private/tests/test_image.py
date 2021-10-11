from cenotaph.basics.base_classes import Image 

img_file_RGB_8 = '../images/peppers.jpg'
img = Image(img_file_RGB_8)
img.print_info()

img.to_greyscale()
img.print_info()