import cv2
import mediapipe as mp
import numpy as np
from random import randint
from engine import BodyEngine, GameEngine, PosesEngine, ActionQueue

renderer = GameEngine(cv2.VideoCapture(0))
body = BodyEngine()
shadow = PosesEngine()
queue = ActionQueue()

shadow_color = (100,100,100)
shadow_thickness = 10
skeleton_color = (0,0,0)
skeleton_thickness = 4
pose_number = 1
new_pose = {}

# Run the game loop
running = True
while running:
    renderer.update()
    body.process_frame(renderer.get_frame())
    renderer.drawPose(body.points, (0,0,0), 5)
    y,x,_ = renderer.get_frame().shape
    valid = False
    if queue.getFirstFromQueue() == 0:
        new_pose = shadow.calculatePose([200,300], x,y, pose_number)
        queue.addToQueue(1)
        queue.forwardQueue()
    elif queue.getFirstFromQueue() == 1:
        valid = shadow.checkPose(body.points)
        if valid:
            queue.addToQueue(2)
            queue.forwardQueue()
    elif queue.getFirstFromQueue() == 2:
        pose_number = randint(1,4)
        queue.addToQueue(0)
        queue.forwardQueue()
    elif queue.getFirstFromQueue() == 3:
        pass

    color = (0,255,0) if valid else (0,0,255)
    renderer.drawPose(new_pose,color,20)


    cv2.imshow("\"Game\"", renderer.get_frame())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

renderer.close()
cv2.destroyAllWindows()
