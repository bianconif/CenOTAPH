from cenotaph.basics.base_classes import Image
from cenotaph.texture.hep.greyscale import LBP

img_file = '../images/peppers.jpg'
img_in = Image(img_file)

lbp_dir = LBP(radius=2, num_peripheral_points = 8)
lbp_cyc = LBP(radius=2, num_peripheral_points = 8, group_action = 'C')
#f_lbp_dir = lbp_dir.get_features(img_in)
f_lbp_cyc = lbp_cyc.get_features(img_in)
print(lbp_dir)
print(lbp_cyc)
a = 0