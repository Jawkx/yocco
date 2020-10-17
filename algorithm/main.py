import numpy as np
import cv2
import time

from find_faces import find_faces, label_faces
from get_landmark import get_landmark, draw_marks
cap = cv2.VideoCapture(0)

new_frame_time = 0
prev_frame_time = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    new_frame_time = time.time()
    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Our operations on the frame come here

    if ret:
        faces_coor, confidence = find_faces(frame, 0.7)
        print(confidence)
        output = label_faces(frame, faces_coor)
        # Display the resulting frame
        for face_coor in faces_coor:
            mark = get_landmark(frame, face_coor)
            output = draw_marks(frame, mark)

        cv2.imshow("display", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
