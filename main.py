import os
import sys
import time
import uuid
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras


def preprocess_frame(frame, bgavg, threshold):
    diff = cv2.absdiff(bgavg.astype("uint8"), frame)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    #thresholded = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,5)
    return thresholded


def _main():
    width, height = 128, 128
    src = cv2.VideoCapture(0 if len(sys.argv) == 1 else sys.argv[1])
    src.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    src.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    win_name = 'preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    #model = keras.models.load_model('bgmodel')
    model = keras.models.load_model('manualmodel85')

    default_text_params = {
            'fontFace': cv2.FONT_HERSHEY_PLAIN,
            'fontScale': 0.7,
            'color': (255, 0, 0),
            'thickness': 1,
            'lineType': cv2.LINE_AA,
    }
   
    frame_rate = 5
    s_time = time.time()
    key = None
    nframe = 0
    bgavg = None
    threshold = 25

    while True:
        c_time = time.time()
        elapsed = c_time - s_time
        has_frame, frame = src.read()
        if not has_frame:
            break
        if elapsed < 1. / frame_rate:
            continue

        nframe += 1

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            nframe = 0
            bgavg = None
            continue
        elif key == 84:
            threshold = max(0, threshold - 1)
        elif key == 82:
            threshold = min(100, threshold + 1)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (width, height))

        if nframe < 30:
            if bgavg is None:
                bgavg = frame.astype(np.float64)
            else:
                cv2.accumulateWeighted(frame, bgavg, 0.2)
            continue

        fps = int(1.0 / elapsed)
        s_time = c_time
        
        strings = []

        strings.append(f'FPS: {fps}')

        strings.append(f'{threshold}')

        frame = preprocess_frame(frame, bgavg, threshold)

        if key == 13:
            cv2.imwrite(os.path.join('manual', uuid.uuid4().hex + '.png'), frame)
        
        ypred = model.predict(np.expand_dims(np.expand_dims(frame, axis=0), axis=3))[0]
        label = ypred.argmax()
        conf = np.exp(ypred[label])/np.exp(ypred).sum()
        #print(np.exp(ypred)/np.exp(ypred).sum())
        strings.append(f'{ypred.argmax()} {conf:.2}')

        cv2.putText(frame, ' | '.join(strings), (10, 10), **default_text_params)
        cv2.imshow(win_name, frame)

    src.release()
    cv2.destroyWindow(win_name)


if __name__ == '__main__':
    _main()
