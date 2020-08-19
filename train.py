import os
import cv2
from PIL import Image
import numpy as np

cascade = cv2.CascadeClassifier("C:/Users/lohiy/PycharmProjects/pythonProject/haarcascade_frontalface_default.xml")

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir,"New folder")
x_train = []
y_train = []
i=0
label_ids = {'emilia clarke': 0, 'kit harington': 1, 'nikolaj': 2, 'peter dinklage': 3}
recognizer = cv2.face.LBPHFaceRecognizer_create()
for root,dirs,files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root,file)
        label = os.path.basename(os.path.dirname(path))
        id = label_ids[label]
        image = Image.open(path).convert("L")
        image_array = np.array(image,"uint8")
        x_train.append(image_array)
        y_train.append(id)

recognizer.train(x_train,np.array(y_train))
recognizer.save("recognizer.yml")