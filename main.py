import os
import sys
import time
import uuid
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras


def attach_text(image, text):
    cv2.putText(image, text, (20, 20),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
    )


def preprocess_frame(frame, bgframes):
    bgsub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    for bgframe in bgframes:
        bgsub.apply(bgframe)
    mask = bgsub.apply(frame)
    #return np.expand_dims(cv2.cvtColor(cv2.bitwise_and(frame, frame, mask=mask), cv2.COLOR_BGR2GRAY), axis=2)
    return np.expand_dims(mask, axis=2)


def _main():
    width, height = 128, 128
    src = cv2.VideoCapture(0 if len(sys.argv) == 1 else sys.argv[1])
    src.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    src.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    win_name = 'preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    #model = keras.models.load_model('bgmodel')
    model = keras.models.load_model('manualmodel')

    default_text_params = {
            'fontFace': cv2.FONT_HERSHEY_PLAIN,
            'fontScale': 0.7,
            'color': (255, 0, 0),
            'thickness': 1,
            'lineType': cv2.LINE_AA,
    }
   
    frame_rate = 7
    bgframes = []
    s_time = time.time()
    key = None
    nframe = 0
    while True:
        nframe += 1
        key = cv2.waitKey(1)
        print(key)
        if key == 27:
            break
        if nframe == 1 or key == 32:
            bgframes = []
            for _ in range(30):
                bgframes.append(cv2.resize(src.read()[1], (width, height)))

        c_time = time.time()
        elapsed = c_time - s_time
        has_frame, frame = src.read()
        if not has_frame:
            break
        if elapsed < 1. / frame_rate:
            continue

        fps = int(1.0 / elapsed)
        s_time = c_time
        
        strings = []

        strings.append(f'FPS: {fps}')
        
        frame = cv2.resize(frame, (width, height))
        frame = preprocess_frame(frame, bgframes)
    
        if key == 13:
            cv2.imwrite(os.path.join('manual', uuid.uuid4().hex + '.png'), frame)
        
        ypred = model.predict(np.expand_dims(frame, axis=0))[0]
        label = ypred.argmax()
        conf = np.exp(ypred[label])/np.exp(ypred).sum()
        print(np.exp(ypred)/np.exp(ypred).sum())

        strings.append(f'{ypred.argmax()} {conf:.2}')

        cv2.putText(frame, ' | '.join(strings), (10, 10), **default_text_params)
        cv2.imshow(win_name, frame)

    src.release()
    cv2.destroyWindow(win_name)


if __name__ == '__main__':
    _main()
