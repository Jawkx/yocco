import cv2

from faced import FaceDetector

from get_landmark import get_square

face_detector = FaceDetector()  # load from faced


def annotate_image(frame, bboxes):
    ret = frame[:]

    img_h, img_w, _ = frame.shape

    for x, y, w, h, p in bboxes:
        cv2.rectangle(ret, (int(x - w/2), int(y - h/2)),
                      (int(x + w/2), int(y + h/2)), (0, 255, 0), 3)
        cv2.putText(ret, str(p), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    return ret


def getFaceSquare(bbox, frame):
    """
    input:
        bbox:
            tuple, 5 parameter :
                x - centerpoint x cooordinate
                y - centerpoint y coordinate
                w - width of bounding box
                h - height of bounding box
                p - percentage of confidence(not used)
        frame:
            np img, unt8, frame from whole camera

    output:
        squared bounding box
    """

    (x, y, w, h, p) = bbox

    if (w == h):
        return frame[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
    if w > h:
        # fat box
        h = w
    else:
        # slim box
        w = h

    return frame[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]


def detect_face(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    bboxes = get_square(face_detector.predict(frame_rgb, 0.5))

    if (len(bboxes) != 0):
        bboxes.sort(key=lambda x: x[4])
        ann_im = annotate_image(frame, bboxes)
        main_face_bbox = bboxes[0]
        face_img = getFaceSquare(main_face_bbox, frame)

        return face_img

    else:
        return frame
