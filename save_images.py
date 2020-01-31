import pandas as pd
import numpy as np

import urllib
import cv2
import sys

if __name__ == "__main__":
    HD = "HD"
    max_images = 80000
    if len(sys.argv) > 1:
        if sys.argv[1] == "non-hd":
            print("running in non-hd mode")
            HD = ""
        elif sys.argv[1].isdigit():
            print("only saving the first " + sys.argv[1] + " images")
            max_images = int(sys.argv[1])

data_uk = pd.DataFrame(pd.read_csv(
    "./data/GBvideos" + HD + ".csv", error_bad_lines=False))
data_us = pd.DataFrame(pd.read_csv(
    "./data/USvideos" + HD + ".csv", error_bad_lines=False))
data = pd.concat([data_us, data_uk])

images = []
thumbnails = data[['thumbnail_link']]
print("found " + str(len(thumbnails)) + " thumbnails")

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
        print(i + 1)
        np.savez("data/images" + HD + "/images_" + str(i - 999) +
                 "_" + str(i + 1) + ".npz", images)
        images = []
    if i + 1 == max_images:
        if len(images) > 0:
            np.savez("data/images" + HD + "/images_" + str(i - len(images) + 1) +
                     "_" + str(i + 1) + ".npz", images)
        exit()
np.savez("data/images" + HD + "/images_" + str(79000) +
         "_" + str(79865) + ".npz", images)
