import cv2
import mediapipe as mp
import numpy as np
from engine import BodyEngine, ShadowEngine
import yaml

renderer = ShadowEngine(cv2.VideoCapture(0))
body = BodyEngine()

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

def landmark2vector(l):
    return np.array([l.x, l.y])
def get_angles(landmarks):
    landmarks = landmarks.landmark
    d = dict()
    for key in body_bones:
        bone = body_bones[key]
        if key == "body":
            start = (landmark2vector(landmarks[bone[0]]) + landmark2vector(landmarks[bone[2]]))/2
            stop = (landmark2vector(landmarks[bone[1]]) + landmark2vector(landmarks[bone[3]]))/2
        else:
            start = landmark2vector(landmarks[bone[0]])
            stop =  landmark2vector(landmarks[bone[1]])
        dr = stop - start
        d[key] = np.arctan2(dr[0], dr[1])
    return d


#pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

with open('landmarks.yaml', 'r') as f:
    landmarks = yaml.safe_load(f)
landmark_names = {v: k for k, v in landmarks.items()}

# https://google.github.io/mediapipe/solutions/pose
connections = {11 : [12,13,23],
               12 : [14, 24],
               13 : [15],
               14 : [16],
               15 : [17, 19],
               16 : [18, 20],
               17 : [19],
               18 : [20],
               19 : [],
               20 : [],
               21 : [],
               22 : [],
               23 : [24, 25],
               24 : [26],
               25 : [27],
               26 : [28],
               27 : [29, 31],
               28 : [30, 32],
               29 : [31],
               30 : [32],
               31 : [],
               32 : [],
               }

def convert_xy(point, screenX,screenY):
    conv = point * np.array((screenX,screenY))
    return conv.astype(int)

# Run the game loop
running = True
while running:
    renderer.update()
    body.process_frame(renderer.get_frame())
    renderer.drawPose(body.points, (0,0,0), 5)
    print(get_angles(result.pose_landmarks))




    cv2.imshow("\"Game\"", renderer.get_frame())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

renderer.close()
cv2.destroyAllWindows()
