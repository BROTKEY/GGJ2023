import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
                          
# https://google.github.io/mediapipe/solutions/pose
connections = {11 : [12,13,23],
               12 : [14, 24],
               13 : [15],
               14 : [12,16],
               15 : [13,19,17],
               16 : [14,18,20],
               17: [15,19],
               18: [16, 20],
               19: [15,17],
               20: [16,18],
               23: [11,25,24],
               24: [23,26,12],
               25: [23,27],
               26:[24,28],
               30: [28,32],
               32: [28],
               27: [29,31],
               31: [29]
               }

def convert_xy(x,y, screenX,screenY):
    return ( np.array((x,y)) * ( np.array((screenX,screenY)) ) )

# Run the game loop
running = True
while running and cap.isOpened():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #BaseImage = np.ones([720,1280,3])
    BaseImage = frame

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(RGB)
    y,x,_ = BaseImage.shape

    drawn_connections = []
    for connection in connections:
        try:
            mark = result.pose_landmarks.landmark[connection]
            mark = convert_xy(mark.x, mark.y,x,y)
            for end in connections[connection]:
                if [mark, end].sort() in drawn_connections:
                    continue
                point = result.pose_landmarks.landmark[end]
                point = convert_xy(point.x, point.y,x,y)
                BaseImage = cv2.line(BaseImage, (int(mark[0]), int(mark[1])), (int(point[0]), int(point[1])), (100, 100, 100), 10)
                BaseImage = cv2.line(BaseImage, (int(mark[0]), int(mark[1])), (int(point[0]), int(point[1])), (0,0,0), 4)
                drawn_connections.append([mark, end].sort())
        except Exception as e:
            print(e)
            continue

    cv2.imshow("dhjsdahdsl", BaseImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()