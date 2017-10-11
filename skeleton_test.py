from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage import util
import numpy as np
from scipy import misc
from skimage import color
from skimage import io
from scipy.misc import toimage

# Invert the horse image
image1 = (color.rgb2gray(io.imread('test_image.png')))
misc.imsave('outfile.jpg', image1)
image = np.invert(image1)
print(image.shape)

# image = np.invert(init_image)

# perform skeletonization
skeleton = skeletonize(image)

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})

ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()