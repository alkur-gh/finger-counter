import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras

def process_frame(frame):
    return cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2GRAY)

def _main():
    model_dir = os.path.join('models', 'dummy-model')
    model = keras.models.load_model(model_dir)
    src = cv2.VideoCapture(0 if len(sys.argv) == 1 else sys.argv[1])
    #src = cv2.VideoCapture('http://192.168.1.139:8080/video')
    win_name = 'preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while cv2.waitKey(1) != 27:
        has_frame, frame = src.read()
        if not has_frame:
            break

        pframe = fgbg.apply(frame)
        pframe = cv2.resize(pframe, (128, 128))
        cv2.imshow(win_name, pframe)
        #label = model.predict(np.expand_dims(
        #    pframe.reshape(*pframe.shape, 1),
        #    axis=0)).argmax()
        #text = "Prediction: " + str(label)
        #cv2.putText(frame, text, (50, 50),
        #        fontFace=cv2.FONT_HERSHEY_PLAIN,
        #        fontScale=2.3,
        #        color=(255, 0, 0),
        #        thickness=2,
        #        lineType=cv2.LINE_AA,
        #)
        #cv2.imshow(win_name, frame)
    src.release()
    cv2.destroyWindow(win_name)


if __name__ == '__main__':
    _main()
