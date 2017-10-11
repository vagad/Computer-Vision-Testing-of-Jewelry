from scipy import misc
import numpy as np
from skimage import color
from skimage import io
from scipy.misc import toimage

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.data import camera
from skimage.color import label2rgb


imaging2 = camera()
print(imaging2.shape)
toimage(imaging2).show()

imaging = data.load('brick.png')
print(imaging.shape)
toimage(imaging).show()

image = misc.imread('test_image.png')
test = image.transpose(2,0,1).reshape(3,-1)
print(image[0][0])
print(image.shape)
print(test.shape)

img = color.rgb2gray(io.imread('test_image.png'));
print(img.shape)

toimage(img).show()