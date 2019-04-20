import numpy as np
import cv2
import pickle

FACE_CASCADE = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')
EYE_CASCADE = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

CAP = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    RET, FRAME = CAP.read()
    GRAY = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)
    FACES = FACE_CASCADE.detectMultiScale(
        GRAY, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in FACES:
        # print(x, y, w, h)
        roi_gray = GRAY[y:y+h, x:x+w]  # (ycord_start, ycord_end)
        roi_color = FRAME[y:y+h, x:x+w]

        # recognize? deep learned model predict
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(FRAME, name, (x, y), font, 1,
                        color, stroke, cv2.LINE_AA)

        img_item = 'my-image.png'
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)  # BGR color 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(FRAME, (x, y), (end_cord_x, end_cord_y), color, stroke)

        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', FRAME)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
CAP.release()
cv2.destroyAllWindows()
