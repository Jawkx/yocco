import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


#model = keras.models.load_model('algorithm\models\pose_model')


def get_square(bboxes):
    bboxesNew = []
    for bbox in bboxes:
        (x, y, w, h, p) = bbox

        if (w == h):
            bboxesNew.append(bbox)
        if w > h:
            # fat box
            h = w
        else:
            w = h

        bboxesNew.append((x, y, w, h, p))

    return bboxesNew


"""
def get_marks(face_img):
    face_img = face_img.astype(np.uint8)
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    predictions = model.signatures["predict"](
        tf.constant([face_img], dtype=tf.uint8))

    marks = np.array(predictions['output']).flatten()[:136]
    marks = np.reshape(marks, (-1, 2))

    marks *= (facebox[2] - facebox[0])
    marks[:, 0] += facebox[0]
    marks[:, 1] += facebox[1]
    marks = marks.astype(np.uint)

    return marks
"""


def draw_marks(face_img, marks, color=(0, 255, 0)):
    for mark in marks:
        cv2.circle(face_img, (mark[0], mark[1]), 2, color, -1, cv2.LINE_AA)

    return face_img
