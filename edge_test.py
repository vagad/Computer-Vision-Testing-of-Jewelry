import numpy as np
import matplotlib.pyplot as plt
import os

from scipy import misc
from skimage import color
from skimage import io
from scipy.misc import toimage
from skimage.filters import roberts, sobel, scharr, prewitt

path1 = 'attachments/'

listing = os.listdir(path1)    
for file in listing:
	init_image = misc.imread(path1 + file)
	toimage(init_image).show()

	image = color.rgb2gray(io.imread(path1 + file));
	edge_roberts = roberts(image)
	edge_sobel = sobel(image)

	print(edge_roberts.shape)
	print(edge_sobel.shape)

	fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True,
	                       figsize=(8, 4))

	ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
	ax[0].set_title('Roberts Edge Detection')

	ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
	ax[1].set_title('Sobel Edge Detection')

	ax[2].imshow(image, cmap=plt.cm.gray)
	ax[2].set_title('Initial Image')

	for a in ax:
	    a.axis('off')

	plt.tight_layout()

	plt.show()