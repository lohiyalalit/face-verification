import os
import cv2
import numpy as np

cascade = cv2.CascadeClassifier("C:/Users/lohiy/OneDrive/Desktop/friends/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer.yml")
img  = cv2.imread('C:/Users/lohiy/PycharmProjects/pythonProject/test/9.jpg')
img = cv2.resize(img,(650,650))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray,scaleFactor=1.32,minNeighbors=5)
label_id = {0:"emilia clarke",1:"kit harington",
            2:"nikolaj",3:"peter dinklage"}

for (x,y,w,h) in faces:
    roi = gray[y:y+h,x:x+w]
    id, conf = recognizer.predict(roi)
    width = x+w
    height = y+h
    color = (0,255,0)
    #if("kit harington"==label_id[id]):
    cv2.rectangle(img,(x,y),(width,height),color,2)
    cv2.putText(img,label_id[id],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)
    #else:
    #    k_size = (w,h)
    #    img[y:y + h, x:x + w] = cv2.blur(img[y:y+h,x:x+w],(10,10))
    #else:
     #   cv2.rectangle(img, (x, y), (width, height), color, -1)
cv2.imshow("img",img)
cv2.imwrite('result_2.jpg',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(id)