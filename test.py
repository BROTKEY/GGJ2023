import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(RGB)

    y,x,_ = frame.shape

    try:
        nose = results.pose_landmarks.landmark[0]
        frame = cv2.circle(frame, (int(nose.x * x), int(nose.y *y)), 3,(0,0,255), 1)
    except:
        continue

    try:
        handleft = results.pose_landmarks.landmark[15]
        frame = cv2.circle(frame, (int(handleft.x * x), int(handleft.y *y)), 3,(0,0,255), 1)
    except:
        continue

    try:
        handright = results.pose_landmarks.landmark[16]
        frame = cv2.circle(frame, (int(handright.x * x), int(handright.y *y)), 3,(0,0,255), 1)
    except:
        continue

    try:
        footright = results.pose_landmarks.landmark[28]
        frame = cv2.circle(frame, (int(footright.x * x), int(footright.y *y)), 3,(0,0,255), 1)
    except:
        continue

    try:
        footleft = results.pose_landmarks.landmark[27]
        frame = cv2.circle(frame, (int(footleft.x * x), int(footleft.y *y)), 3,(0,0,255), 1)
    except:
        continue

    cv2.imshow("dhjsdahdsl", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
