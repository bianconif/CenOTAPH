from cenotaph.basics.digital_circle import *

radius = 1
n = 2
original_circle = digital_circle(radius, n)

n_points = 8
alpha = 0
downsapled_circle = downsample_circle(original_circle, n_points, alpha)