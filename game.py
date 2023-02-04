import cv2
import mediapipe as mp
import numpy as np
from engine import BodyEngine, GameEngine, PosesEngine

renderer = GameEngine(cv2.VideoCapture(0))
body = BodyEngine()
shadow = PosesEngine()

shadow_color = (100,100,100)
shadow_thickness = 10
skeleton_color = (0,0,0)
skeleton_thickness = 4

body_bones = dict({
    "lower_arm_l" : [13, 15],
    "lower_arm_r" : [14, 16],
    "upper_arm_l" : [11 ,13],
    "upper_arm_r" : [12 ,14],
    "upper_leg_l" : [23, 25],
    "upper_leg_r" : [24, 26],
    "lower_leg_l" : [25, 27],
    "lower_leg_r" : [26, 28],
    "body": [23, 11 , 24, 12],
    })

def get_angles(landmarks):
    d = dict()
    for key in body_bones:
        bone = body_bones[key]
        if key == "body":
            start = (landmarks[bone[0]] + landmarks[bone[2]])/2
            stop = (landmarks[bone[1]] + landmarks[bone[3]])/2
        else:
            start = landmarks[bone[0]]
            stop =  landmarks[bone[1]]
        dr = stop - start
        d[key] = np.arctan2(dr[0], dr[1])
    return d

duck = cv2.imread("debugger.jpeg")

# Run the game loop
running = True
while running:
    renderer.update()
    body.process_frame(renderer.get_frame())
    renderer.drawImage(duck, (320,230), (64,64))
    renderer.drawPose(body.points, (0,0,0), 5)
    y,x,_ = renderer.get_frame().shape
    s = shadow.calculatePose([200,300], x,y, int(1))
    renderer.drawPose(s,(100,100,100),20)


    cv2.imshow("\"Game\"", renderer.get_frame())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

renderer.close()
cv2.destroyAllWindows()
