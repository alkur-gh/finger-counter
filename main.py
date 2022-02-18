import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
import mediapipe as mp


def process_frame(frame):
    return cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2GRAY)


def attach_text(image, text):
    cv2.putText(image, text, (20, 20),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
    )


def extract_features(results):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True) as hands:
        if not results.multi_hand_world_landmarks:
            return None

        hand_landmark = results.multi_hand_world_landmarks[0] 
        features = []
        for v in hand_landmark.landmark:
            features.append(v.x)
            features.append(v.y)

        return features


def _main():
    model_dir = os.path.join('models', 'dummy-model')
    model = keras.models.load_model(model_dir)
    src = cv2.VideoCapture(0 if len(sys.argv) == 1 else sys.argv[1])
    win_name = 'preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    mirror = True

    default_text_params = {
            'fontFace': cv2.FONT_HERSHEY_PLAIN,
            'fontScale': 1,
            'color': (255, 0, 0),
            'thickness': 1,
            'lineType': cv2.LINE_AA,
    }

    frame_rate = 10
    s_time = time.time()
    with mp_hands.Hands(model_complexity=0) as hands:
        while cv2.waitKey(1) != 27:
            c_time = time.time()
            elapsed = c_time - s_time
            has_frame, frame = src.read()
            if elapsed < 1. / frame_rate:
                continue

            fps = int(1. / (c_time - s_time))
            s_time = c_time

            if not has_frame:
                break

            strings = []

            strings.append(f'FPS: {fps}')

            #frame = cv2.resize(frame, (256, 256))
        
            h, w, c = frame.shape
            results = hands.process(frame)
            if results.multi_hand_landmarks:
                for hand_landmark, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    is_left = handedness.classification[0].label == ('Right' if mirror else 'Left')
                    cv2.putText(frame, ['RIGHT', 'LEFT'][is_left],
                            (int(w * hand_landmark.landmark[0].x) + 20, int(h * hand_landmark.landmark[0].y)),
                            **default_text_params)
                    mp_drawing.draw_landmarks(
                            frame,
                            hand_landmark,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                    )

                features = extract_features(results)
                if features:
                    label = model.predict(np.array([features])).argmax()
                    strings.append(f'Label: {label}')

            cv2.putText(frame, ' | '.join(strings), (20, 20), **default_text_params)
            cv2.imshow(win_name, frame)

    src.release()
    cv2.destroyWindow(win_name)


if __name__ == '__main__':
    _main()
