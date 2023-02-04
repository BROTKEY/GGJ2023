import cv2
import mediapipe as mp
import numpy as np
import yaml
#from engine import BodyEngine

cap = cv2.VideoCapture(0)

# body = BodyEngine()

mp_pose = mp.solutions.pose

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


pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

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

def convert_xy(x,y, screenX,screenY):
    return ( np.array((x,y)) * ( np.array((screenX,screenY)) ) )

# Run the game loop
running = True
while running and cap.isOpened():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #base_image = np.ones([720,1280,3])
    base_image = frame

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(RGB)
    y,x,_ = base_image.shape

    one = [0,0]
    two = [0,0]
    three = [0,0]
    four = [0,0]
    
    drawn_connections = []
    for connection in connections:
        if connection is []:
            continue
        try:
            mark = result.pose_landmarks.landmark[connection]

            if connection == 11:
                one = np.array([mark.x, mark.y])
            
            if connection == 12:
                two = np.array([mark.x, mark.y])

            if connection == body_bones[""]:
                three = np.array([mark.x, mark.y])

            if connection == 24:
                four = np.array([mark.x, mark.y])
                
            mark = convert_xy(mark.x, mark.y, x, y)
            for end in connections[connection]:
                point = result.pose_landmarks.landmark[end]
                point = convert_xy(point.x, point.y, x, y)
                base_image = cv2.line(base_image, (int(mark[0]), int(mark[1])), (int(point[0]), int(point[1])), shadow_color, shadow_thickness)
                base_image = cv2.line(base_image, (int(mark[0]), int(mark[1])), (int(point[0]), int(point[1])), skeleton_color, skeleton_thickness)
        except Exception as e:
            print(e)
            continue

    print("ratio:", np.linalg.norm((one-two)) / np.linalg.norm((three-four)))

    cv2.imshow("dhjsdahdsl", base_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
