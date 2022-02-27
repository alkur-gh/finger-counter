import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input


def attach_text(image, text):
    cv2.putText(image, text, (20, 20),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
    )


def _main():
    width, height = 300, 300
    src = cv2.VideoCapture(0 if len(sys.argv) == 1 else sys.argv[1])
    src.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    src.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    win_name = 'preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    default_text_params = {
            'fontFace': cv2.FONT_HERSHEY_PLAIN,
            'fontScale': 1,
            'color': (255, 0, 0),
            'thickness': 1,
            'lineType': cv2.LINE_AA,
    }

    base_model = VGG16(weights='imagenet', include_top=False)
    model = keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer('block2_conv2').output)

    s_time = time.time()
    while cv2.waitKey(1) != 27:
        c_time = time.time()
        elapsed = c_time - s_time
        has_frame, frame = src.read()
        if not has_frame:
            break
        fps = int(1.0 / elapsed)
        s_time = c_time

        strings = []

        strings.append(f'FPS: {fps}')
        
        x = preprocess_input(np.expand_dims(frame, axis=0))
        features = model.predict(x)
        print(features.shape)

        cv2.putText(frame, ' | '.join(strings), (20, 20), **default_text_params)
        cv2.imshow(win_name, frame)

    src.release()
    cv2.destroyWindow(win_name)


if __name__ == '__main__':
    _main()
