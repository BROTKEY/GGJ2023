import cv2
import mediapipe as mp
import numpy as np
from random import randint
from engine import BodyEngine, GameEngine, PosesEngine, ActionQueue
from datetime import datetime

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
score=0
valid_time = 0

last_time = datetime.now()

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
        now = datetime.now()
        timedelta = (now - last_time).total_seconds()
        if valid:
            valid_time += 100 * timedelta
            if valid_time > 255:
                valid_time = 0
                score += 1
                queue.addToQueue(2)
                queue.forwardQueue()
        else:
            if valid_time < 0:
                valid_time = 0
            else:
                valid_time -= 150 * timedelta
        last_time = now
    elif queue.getFirstFromQueue() == 2:
        pose_number = randint(1,4)
        queue.addToQueue(0)
        queue.forwardQueue()
    elif queue.getFirstFromQueue() == 3:
        pass

    color = (0,255,255-valid_time) if valid else (0,128,255-valid_time)
    renderer.drawPose(new_pose,color,20)


    cv2.imshow("\"Game\"", renderer.get_frame())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

renderer.close()
cv2.destroyAllWindows()
