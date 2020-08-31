# Computer Vision Project
# Qazi Umer Jamil
# RIME 19, NUST Regn No 317920

import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0) # To capture webcam feed
#cap = cv2.VideoCapture('http://192.168.10.6:8080/video') # to capture feed from an IP Camera
#cap = cv2.VideoCapture('/Users/Umer/Desktop/CV_ass/zoom_1.mp4') # To read video file

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def calculate_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

#    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
 #   ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def draw_facial_parts(facial_landmarks):
    # Draw circle on left and right eye
    left_eye_points = [36, 37, 38, 39, 40, 41]

    x_point_for_circle = int((landmarks.part(left_eye_points[0]).x + landmarks.part(left_eye_points[3]).x) / 2)
    y_point_for_circle = int((landmarks.part(left_eye_points[0]).y + landmarks.part(left_eye_points[3]).y) / 2)

    cv2.circle(frame, (x_point_for_circle, y_point_for_circle), 9, (0, 0, 255))

    right_eye_points = [42, 43, 44, 45, 46, 47]

    x_point_for_circle = int((landmarks.part(right_eye_points[0]).x + landmarks.part(right_eye_points[3]).x) / 2)
    y_point_for_circle = int((landmarks.part(right_eye_points[0]).y + landmarks.part(right_eye_points[3]).y) / 2)

    cv2.circle(frame, (x_point_for_circle, y_point_for_circle), 9, (0, 0, 255))

    # Draw polygon on Nose
    nose_points = [ 27, 28, 29, 30, 31, 32, 33, 34, 35, 30]
    nose_region = np.array([(facial_landmarks.part(nose_points[0]).x, facial_landmarks.part(nose_points[0]).y),
                            (facial_landmarks.part(nose_points[1]).x, facial_landmarks.part(nose_points[1]).y),
                            (facial_landmarks.part(nose_points[2]).x, facial_landmarks.part(nose_points[2]).y),
                            (facial_landmarks.part(nose_points[3]).x, facial_landmarks.part(nose_points[3]).y),
                            (facial_landmarks.part(nose_points[4]).x, facial_landmarks.part(nose_points[4]).y),
                            (facial_landmarks.part(nose_points[5]).x, facial_landmarks.part(nose_points[5]).y),
                            (facial_landmarks.part(nose_points[6]).x, facial_landmarks.part(nose_points[6]).y),
                            (facial_landmarks.part(nose_points[7]).x, facial_landmarks.part(nose_points[7]).y),
                            (facial_landmarks.part(nose_points[8]).x, facial_landmarks.part(nose_points[8]).y)], np.int32)
    cv2.polylines(frame, [nose_region], True, (0, 0, 255), 2)

    #cv2.putText(frame, "Nose", int(facial_landmarks.part(nose_points[7]).x), int(facial_landmarks.part(nose_points[7]).y), font, 2, (0, 0, 255), 3)

    # Draw polygon on Mouth
    mount_points = [ 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,58,59,48]
    nose_region = np.array([(facial_landmarks.part(mount_points[0]).x, facial_landmarks.part(mount_points[0]).y),
                            (facial_landmarks.part(mount_points[1]).x, facial_landmarks.part(mount_points[1]).y),
                            (facial_landmarks.part(mount_points[2]).x, facial_landmarks.part(mount_points[2]).y),
                            (facial_landmarks.part(mount_points[3]).x, facial_landmarks.part(mount_points[3]).y),
                            (facial_landmarks.part(mount_points[4]).x, facial_landmarks.part(mount_points[4]).y),
                            (facial_landmarks.part(mount_points[5]).x, facial_landmarks.part(mount_points[5]).y),
                            (facial_landmarks.part(mount_points[6]).x, facial_landmarks.part(mount_points[6]).y),
                            (facial_landmarks.part(mount_points[7]).x, facial_landmarks.part(mount_points[7]).y),
                            (facial_landmarks.part(mount_points[8]).x, facial_landmarks.part(mount_points[8]).y),
                            (facial_landmarks.part(mount_points[9]).x, facial_landmarks.part(mount_points[9]).y),
                            (facial_landmarks.part(mount_points[10]).x, facial_landmarks.part(mount_points[10]).y),
                            (facial_landmarks.part(mount_points[11]).x, facial_landmarks.part(mount_points[11]).y),
                            (facial_landmarks.part(mount_points[12]).x, facial_landmarks.part(mount_points[12]).y)
                            ], np.int32)
    cv2.polylines(frame, [nose_region], True, (0, 0, 255), 2)

    # Draw polygon on Jaw
    jaw_points = [6, 7, 8, 9, 10]
    left_eye_region = np.array([(facial_landmarks.part(jaw_points[0]).x, facial_landmarks.part(jaw_points[0]).y),
                                (facial_landmarks.part(jaw_points[1]).x, facial_landmarks.part(jaw_points[1]).y),
                                (facial_landmarks.part(jaw_points[2]).x, facial_landmarks.part(jaw_points[2]).y),
                                (facial_landmarks.part(jaw_points[3]).x, facial_landmarks.part(jaw_points[3]).y),
                                (facial_landmarks.part(jaw_points[4]).x, facial_landmarks.part(jaw_points[4]).y)], np.int32)
    cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    #cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    gray_eye1 = cv2.resize(gray_eye, None, fx=5, fy=5)

    #cv2.imshow("gray_eye1", gray_eye1)

    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape

    kernel = np.ones((5, 5), np.uint8)

    erodded_eye = cv2.erode(threshold_eye, kernel, iterations=2)
    #erodded_eye = cv2.resize(erodded_eye, None, fx=5, fy=5)
    cv2.imshow("erodded_eye", erodded_eye)

    dilated_eye = cv2.dilate(erodded_eye, kernel, iterations=4)
    cv2.imshow("dilated_eye", dilated_eye)

    medianblur_eye = cv2.medianBlur(dilated_eye, 3)
    cv2.imshow("medianblur_eye", medianblur_eye)
    threshold_eye=medianblur_eye

    upper_side_threshold = threshold_eye[0: int(height/2), :]
    upper_side_threshold = cv2.resize(upper_side_threshold, None, fx=15, fy=15)
    cv2.imshow("upper_side_threshold", upper_side_threshold)
    upper_side_threshold = cv2.countNonZero(upper_side_threshold)

    bottom_side_threshold = threshold_eye[int(height/2): height, :]
    bottom_side_threshold = cv2.resize(bottom_side_threshold, None, fx=15, fy=15)
    cv2.imshow("bottom_side_threshold", bottom_side_threshold)
    bottom_side_threshold = cv2.countNonZero(bottom_side_threshold)

    if bottom_side_threshold == 0:
        bottom_side_threshold = 1
    elif upper_side_threshold == 0:
        upper_side_threshold = 1

    gaze_ratio_vertical = upper_side_threshold / bottom_side_threshold

    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio_horizental = 1
    elif right_side_white == 0:
        gaze_ratio_horizental = 5
    else:
        gaze_ratio_horizental = left_side_white / right_side_white

    return gaze_ratio_horizental, gaze_ratio_vertical

while True:
    # Reading video frames
    _, frame = cap.read()
    #new_frame = np.zeros((500, 500, 3), np.uint8) # TO BE REMOVED
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray",gray)

    #Applying CLAHE technique
    clahe = cv2.createCLAHE(clipLimit=5)
    gray = clahe.apply(gray) #+ 30
    #cv2.imshow("clahe",gray)

    font = cv2.FONT_HERSHEY_DUPLEX

    faces = detector(gray)
    # Loop through detected faces
    for face in faces:
        # Draw a box around the face
        left, top = face.left(), face.top()
        right, bottom = face.right(), face.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        landmarks = predictor(gray, face)

        # Detect blinking (eyes open and close)
        left_eye_ratio = calculate_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = calculate_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio_avg = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio_avg > 5.7:
            cv2.putText(frame, "-> Blinking", (50, 100), font, 2, (0, 0, 255), 3)

        # Gaze detection
        gaze_ratio_left_eye_h, gaze_ratio_left_eye_v = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye_h, gaze_ratio_right_eye_v = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio_h = (gaze_ratio_right_eye_h + gaze_ratio_left_eye_h) / 2
        gaze_ratio_v = (gaze_ratio_right_eye_v + gaze_ratio_left_eye_v) / 2

        print("Gaze  Vertical: ", gaze_ratio_v)
        print("Gaze  horizontal: ", gaze_ratio_h)

        if gaze_ratio_h <= 1:
            cv2.putText(frame, "-> Looking Right", (50, 150), font, 2, (0, 0, 255), 3)
            print("Looking Right")
        elif 1 < gaze_ratio_h < 1.7:
            cv2.putText(frame, "-> Looking at Center", (50, 150), font, 2, (0, 0, 255), 3)
            print("Looking at Center")
        else:
            #new_frame[:] = (255, 0, 0)
            cv2.putText(frame, "-> Looking Left", (50, 150), font, 2, (0, 0, 255), 3)
            print("Looking Left")

        if gaze_ratio_v <= 0.8:
            cv2.putText(frame, "-> Looking Down", (50, 200), font, 2, (0, 0, 255), 3)
            print("Looking Down", gaze_ratio_v)
        else:
            cv2.putText(frame, "-> Looking Up", (50, 200), font, 2, (0, 0, 255), 3)
            print("Looking Up", gaze_ratio_v)


        # Drawing facial parts
        draw_facial_parts(landmarks)

    cv2.imshow("Input Video", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()