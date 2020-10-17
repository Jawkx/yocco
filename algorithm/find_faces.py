import cv2
import numpy as np

from faced import FaceDetector


from get_landmark import get_landmark

face_detector = FaceDetector()  # load from faced


def getFaceCoordinate(bboxes):
    """
    Return bbox from faced to face coordinate in ( x , y , x1 , y1 ) format and confidence

    Parameter
    -----------
    bboxes: tuple
        bbox from faced's predict. ( x , y , w , h , c )

    Returns
    -----------
    coordinate: np array
        [ x , y , x1 , y1 ]
    confidence: float
        0.00 ~ 1.00
    """

    converted_bboxes = []
    c = 0
    for bbox in bboxes:
        (x, y, w, h, c) = bbox

        converted_bboxes.append([
            int(x-w/2),
            int(y-h/2),
            int(x+w/2),
            int(y+h/2)
        ])

    return converted_bboxes, c


def find_faces(img, thresh):
    """
        find the faces in the image

        Parameters
        ----------
        img: np.uint8
            the image where faces is to be located
        thresh: float
            minimum confidence to be recognized as "face"


        Returns
        -----------
        faces: array of array[ [ ] , [ ] , ... ]
            array containing coordinates of face
        confidence: float
            confidence that it is a face
    """

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bboxes = face_detector.predict(img_rgb, thresh)
    """
        facedetector.predict will return 5 value
            x: x coordinate of midpoint
            y: y coordinate of midpoint
            w: width of the box
            h: height of the box
            c: confidence that it is a face
    """

    faces, confidence = getFaceCoordinate(bboxes)

    return faces, confidence


def label_faces(img, bboxes):
    """
        draw a bounding box on the faces

        Parameters
        ----------
        img: uint8
            image that is used to get bboxes and image that label to be drawn on
        bboxes: array
            array of face bboxes

        Return
        ----------
        img: uint8
            image that bounding box had been drawn
    """

    for bbox in bboxes:
        img = cv2.rectangle(
            img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
    return img
