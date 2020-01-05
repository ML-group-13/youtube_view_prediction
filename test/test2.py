import pandas as pd
import numpy as np
from PIL import Image

import urllib
import cv2

from PIL import Image

data = np.load('images_0_1000.npz',allow_pickle=True)
#print(data.files)

images = data['arr_0']

image = images[27]

cv2.imshow("Faces found", image)
cv2.waitKey(0)