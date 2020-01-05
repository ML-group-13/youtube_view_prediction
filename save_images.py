import pandas as pd
import numpy as np

import urllib
import cv2

data_uk = pd.DataFrame(pd.read_csv(
    "./data/GBvideos.csv", error_bad_lines=False))
data_us = pd.DataFrame(pd.read_csv(
    "./data/USvideos.csv", error_bad_lines=False))
data = pd.concat([data_us, data_uk])

images = []
thumbnails = data[['thumbnail_link']]
print(len(thumbnails))

for i in range(0, len(thumbnails)):
    thumbnail = thumbnails.values[i][0]
    try:
        resp = urllib.request.urlopen(thumbnail)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        images.append(image)
    except:
        images.append('')
    if len(images) == 1000:
        print(i)
        np.savez("data/images/images_" + str(i - 999) +
                 "_" + str(i + 1) + ".npz", images)
        images = []
np.savez("data/images/images_" + str(79000) +
         "_" + str(79865) + ".npz", images)
