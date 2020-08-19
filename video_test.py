import cv2

cap = cv2.VideoCapture("C:/Users/lohiy/PycharmProjects/pythonProject/test/test_video.mp4")
cascade_classifier = cv2.CascadeClassifier("C:/Users/lohiy/OneDrive/Desktop/friends/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer.yml")
label_id = {0:"emilia clarke",1:"kit harington",
            2:"nikolaj",3:"peter dinklage"}

while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (500, 500))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detect_faces = cascade_classifier.detectMultiScale(frame,scaleFactor=1.32,minNeighbors=2)
    for (x, y, w, h) in detect_faces:
        roi = gray[y:y+h, x:x+w]
        id, conf = recognizer.predict(roi)
        width = x + w
        height = y + h
        color = (255, 0, 0)
        if ("emilia clarke" == label_id[id]):
            cv2.rectangle(frame, (x, y), (width, height), color, 2)
            cv2.putText(frame, label_id[id], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        else:
            k_size = (w, h)
            frame[y:y + h, x:x + w] = cv2.blur(frame[y:y + h, x:x + w], (10, 10))
        cv2.imshow("test_video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()