import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np


cap = cv2.VideoCapture(0)
photo = cv2.imread("jesus.jpg")
#ap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)                  #if need to boost fps
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
detector = HandDetector(detectionCon=0.8, maxHands=2)
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]
def distance(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return math.hypot(x2 - x1, y2 - y1)

def count_fingers(lmList):
    fingers = []

    thumb_tip_dist = distance(lmList[4], lmList[17])
    thumb_joint_dist = distance(lmList[3], lmList[17])

    if thumb_tip_dist > thumb_joint_dist:
        fingers.append(1)
    else:
        fingers.append(0)

    tip_ids = [8, 12, 16, 20]
    for tip_id in tip_ids:            # 4 other fingers
        finger_base = distance(lmList[tip_id - 2], lmList[0])
        finger_tip = distance(lmList[tip_id], lmList[0])
        if finger_tip > finger_base:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)
    
    if hands:
        finger_list = []
        finger_position = []
        wrists = []
        lmLists = []

        for hand in hands:
            lmList = hand["lmList"]
            lmLists.append(lmList)
        
            for start_idx, end_idx in connections:                        # func for bones
                x1, y1, z1 = lmList[start_idx]
                x2, y2, z2 = lmList[end_idx]
                cv2.line(img, (x1, y1), (x2, y2), (214, 170, 28), 3)
            for point in lmList:
                x, y, z = point
                cv2.circle(img, (x, y), 8, (255, 255, 0),cv2.FILLED)

            fingers = count_fingers(lmList)
            finger_list.append(fingers)
            wrists.append(lmList[0][:2])

        h, w, c = img.shape
        if len(finger_list) == 1:
            cv2.putText(img, f"[{finger_list[0][0]},{finger_list[0][1]},{finger_list[0][2]},{finger_list[0][3]},{finger_list[0][4]}]", (wrists[0][0] - 150, wrists[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (105, 255, 70), 3)
            if finger_list[0] == [1,0,1,0,0]:
                break
        elif len(finger_list) == 2:
            cv2.putText(img, f"[{finger_list[0][0]},{finger_list[0][1]},{finger_list[0][2]},{finger_list[0][3]},{finger_list[0][4]}]", (wrists[0][0] - 150, wrists[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (105, 255, 70), 3)
            cv2.putText(img, f"[{finger_list[1][0]},{finger_list[1][1]},{finger_list[1][2]},{finger_list[1][3]},{finger_list[1][4]}]", (wrists[1][0] - 150, wrists[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (105, 255, 70), 3)
            if finger_list[0] == [1,1,0,0,0] and finger_list[1] == [1,1,0,0,0]:
                pt1 = lmLists[0][8][:2]   
                pt2 = lmLists[0][4][:2]   
                pt3 = lmLists[1][4][:2]   
                pt4 = lmLists[1][8][:2]   
                points = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
                cv2.fillPoly(img, [points], (0, 0, 0))
                center_x = (pt1[0] + pt2[0] + pt3[0] + pt4[0]) // 4
                center_y = (pt1[1] + pt2[1] + pt3[1] + pt4[1]) // 4
            elif finger_list[0] == [1,1,1,0,0] and finger_list[1] == [1,1,1,0,0]:
                pt1 = lmLists[0][8][:2]   # указательный, рука 0
                pt2 = lmLists[0][4][:2]   # большой, рука 0
                pt3 = lmLists[1][4][:2]   # большой, рука 1
                pt4 = lmLists[1][8][:2]   # указательный, рука 1
                points = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
                cv2.fillPoly(img, [points], (214, 170, 28))
                center_x = (pt1[0] + pt2[0] + pt3[0] + pt4[0]) // 4
                center_y = (pt1[1] + pt2[1] + pt3[1] + pt4[1]) // 4
            elif finger_list[0] == [1,1,1,1,1] and finger_list[1] == [1,1,1,1,1]:
                pt1 = lmLists[0][8][:2]   # указательный, рука 0
                pt2 = lmLists[0][4][:2]   # большой, рука 0
                pt3 = lmLists[1][4][:2]   # большой, рука 1
                pt4 = lmLists[1][8][:2]   # указательный, рука 1
                points = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
                cv2.fillPoly(img, [points], (255, 255, 255))
                center_x = (pt1[0] + pt2[0] + pt3[0] + pt4[0]) // 4
                center_y = (pt1[1] + pt2[1] + pt3[1] + pt4[1]) // 4
            elif finger_list[0] == [1,0,0,0,0] and finger_list[1] == [1,0,0,0,0]:
                photo = cv2.imread("taks.jpg")
                cv2.imshow("TAKSAAA", photo)
            else:
                cv2.destroyWindow("TAKSAAA")
        else:
            cv2.destroyWindow("TAKSAAA")
    else:
        cv2.destroyWindow("TAKSAAA")

    cv2.imshow("Webcam", img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
