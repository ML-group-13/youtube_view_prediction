from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
from skimage.color import rgb2gray
from collections import Counter
import cv2
import pandas as pd
import urllib
from progress.bar import ChargingBar

model = load_model("feature_extraction\image_feature_extraction\model_v6_23.hdf5")
emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

class ImageFeatureExtractor:

    def extractImageFeatures(self, data):
        image_amount_of_faces = []
        image_text = []
        image_emotions_angry = []
        image_emotions_sad = []
        image_emotions_neutral = []
        image_emotions_disgust = []
        image_emotions_surprise = []
        image_emotions_fear = []
        image_emotions_happy = []
        bar = ChargingBar('Processing Images:\t\t', max=len(data['images']))
        # For all the images in the dataset.
        for idx, image in data['images'].items():
            bar.next()
            image_amount_of_faces.append(self.detect_faces_amount(image))
            image_text.append(self.text_detect(image))
            emotions = self.detect_emotions(image)
            cnt = Counter()
            # Count the amount of emotions present in the image.
            for word in emotions:
                cnt[word] += 1
            image_emotions_angry.append(cnt['Angry'])
            image_emotions_sad.append(cnt['Sad'])
            image_emotions_neutral.append(cnt['Neutral'])
            image_emotions_disgust.append(cnt['Disgust'])
            image_emotions_surprise.append(cnt['Surprise'])
            image_emotions_fear.append(cnt['Fear'])
            image_emotions_happy.append(cnt['Happy'])
        bar.finish()
        data['image_amount_of_faces'] = image_amount_of_faces
        data['image_text'] = image_text
        data['image_emotions_angry'] = image_emotions_angry
        data['image_emotions_sad'] = image_emotions_sad
        data['image_emotions_neutral'] = image_emotions_neutral
        data['image_emotions_disgust'] = image_emotions_disgust
        data['image_emotions_surprise'] = image_emotions_surprise
        data['image_emotions_fear'] = image_emotions_fear
        data['image_emotions_happy'] = image_emotions_happy
        return data

    # Based on the facial_recognition library by Ageitgey: https://github.com/ageitgey/face_recognition
    def detect_faces_amount(self, image):
        img = Image.fromarray(image, 'RGB')
        pic = np.array(img)
        face_locations = face_recognition.face_locations(pic)
        return len(face_locations)

    # Based on the algorithm by qzane: https://github.com/qzane/text-detection
    def text_detect(self, img, ele_size=(8,2)): #
        # Copyright (c) 2015 qzane
        # All rights reserved.
        
        if len(img.shape)==3:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_sobel = cv2.Sobel(img,cv2.CV_8U,1,0)
        img_threshold = cv2.threshold(img_sobel,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        element = cv2.getStructuringElement(cv2.MORPH_RECT,ele_size)
        img_threshold = cv2.morphologyEx(img_threshold[1],cv2.MORPH_CLOSE,element)
        res = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cv2.__version__.split(".")[0] == '3':
            _, contours, hierarchy = res
        else:
            contours, hierarchy = res
        Rect = [cv2.boundingRect(i) for i in contours if i.shape[0]>100]
        RectP = [(int(i[0]-i[2]*0.08),int(i[1]-i[3]*0.08),int(i[0]+i[2]*1.1),int(i[1]+i[3]*1.1)) for i in Rect]
        return len(RectP)

    # Based on the model by Priya-Dwivedi: https://github.com/priya-dwivedi/face_and_emotion_detection
    def detect_emotions(self, image):
        emotions = []
        img = Image.fromarray(image, 'RGB')
        pic = np.array(img)
        face_locations = face_recognition.face_locations(pic)
        if(len(face_locations) > 0):
            face_images = {}
            for j in range(len(face_locations)):
                # Crop and resize the image so that it will fit with the model.
                top, right, bottom, left = face_locations[j]
                face_images[j] = pic[top:bottom, left:right]
                face_image = cv2.resize(face_images[j], (48,48))
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
                # Compare the current face to the model.
                predicted_class = np.argmax(model.predict(face_image))
                label_map = dict((v,k) for k,v in emotion_dict.items()) 
                predicted_label = label_map[predicted_class]
                emotions.append(predicted_label)
        return emotions
