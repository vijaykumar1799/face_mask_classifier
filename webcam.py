import os
import cv2
import pickle
import imutils
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def load_history_file(history_filename):
    pickle_in = open(history_filename, 'rb')
    saved_history = pickle.load(pickle_in)
    return saved_history


def get_faces_coords(img, classifier):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.05, 10)
    return faces


def get_prediction(face, model):
    img = cv2.resize(face, (128, 128), interpolation=cv2.INTER_CUBIC)
    return model.predict(np.expand_dims(img, axis=0))


def main():
    classes = ['mask_weared_incorrect', 'with_mask', 'without_mask']
    model = load_model('./mask_status.h5')
    cascade_classifier = cv2.CascadeClassifier('./cascade_classifier/haarcascade_frontalface_alt2.xml')

    '''cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        face_coords = get_faces_coords(frame, classifier=cascade_classifier)
        for face in face_coords:
            x, y, w, h = face
            roi = frame[y:y+h, x:x+w] / 255.0
            cv2.imshow('roi', roi)
            prediction = get_prediction(face=roi, model=model)
            print(np.argmax(prediction))
            prediction = classes[np.argmax(prediction)]
            print(prediction)

            if prediction == 'with_mask':
                color = (0, 255, 0)
            elif prediction == 'mask_weared_incorrect':
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=2)
            cv2.putText(img=frame, text=prediction, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=color, thickness=2)

        cv2.imshow('webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()'''


    for img in os.listdir('./Test/'):
        img_path = './Test/' + img
        img = cv2.imread(img_path)
        face_coords = get_faces_coords(img=img, classifier=cascade_classifier)
        for face in face_coords:
            x, y, w, h = face
            roi = img[y:y+h, x:x+w] / 255
            prediction = get_prediction(face=roi, model=model)
            prediction = classes[np.argmax(prediction)]
            if prediction == 'with_mask':
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(img, pt1=(x, y), pt2=(x+w, y+h), color=color, thickness=2)
            cv2.putText(img=img, text=prediction, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=2)

        plt.imshow(imutils.opencv2matplotlib(img))
        plt.show()


if __name__ == '__main__':
    main()
