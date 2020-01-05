import pandas as pd
import numpy as np
from PIL import Image

import urllib
import cv2

from PIL import Image

data = np.load('images_0_1000.npz',allow_pickle=True)
#print(data.files)

images = data['arr_0']
count = 0

for i in range(len(images)):
	img = Image.fromarray(images[i], 'RGB')
	img.save('my.png')
	#img.show()

	imagePath = 'my.png'
	cascPath = 'haarcascade_frontalface_default.xml'
	
	faceCascade = cv2.CascadeClassifier(cascPath)

	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	
	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(1, 1),
		#flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	#print ("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
	#cv2.imshow("Faces found", image)
	#cv2.waitKey(0)
	
	if len(faces) > 0:
		count += 1
		
	print(i)
	
	img = []
		
print(count)
	
