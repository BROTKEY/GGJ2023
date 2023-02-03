import cv2
import mediapipe as mp

print(cv2.__file__)
cap = cv2.VideoCapture('/dev/video0')

drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

while cap.isOpened():
    _, frame = cap.read()

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(RGB)
    print(results.pose_landmarks)

    drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("dhjsdahdsl", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
