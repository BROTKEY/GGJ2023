import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose

shadow_color = (100,100,100)
shadow_thickness = 10
skeleton_color = (0,0,0)
skeleton_thickness = 4

pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

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

    drawn_connections = []
    for connection in connections:
        if connection is []:
            continue
        try:
            mark = result.pose_landmarks.landmark[connection]
            mark = convert_xy(mark.x, mark.y, x, y)
            for end in connections[connection]:
                point = result.pose_landmarks.landmark[end]
                point = convert_xy(point.x, point.y, x, y)
                base_image = cv2.line(base_image, (int(mark[0]), int(mark[1])), (int(point[0]), int(point[1])), shadow_color, shadow_thickness)
                base_image = cv2.line(base_image, (int(mark[0]), int(mark[1])), (int(point[0]), int(point[1])), skeleton_color, skeleton_thickness)
        except Exception as e:
            print(e)
            continue

    cv2.imshow("dhjsdahdsl", base_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
