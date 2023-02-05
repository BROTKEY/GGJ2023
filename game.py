import cv2
import mediapipe as mp
import numpy as np
from random import randint
from engine import BodyEngine, GameEngine, PosesEngine, Camera
from engine import BodyEngine, GameEngine, PosesEngine, ActionQueue
from datetime import datetime
import random, yaml

background = cv2.imread("./img/background.png")
cam = Camera()
renderer = GameEngine((1280,720), background, cam.shape)
body = BodyEngine()
shadow = PosesEngine()
queue = ActionQueue()

shadow_color = (100,100,100)
shadow_thickness = 10
skeleton_color = (0,0,0)
skeleton_thickness = 4
poses_avail = list(yaml.load(open("poses.yaml","r"), Loader=yaml.FullLoader).keys())
pose_number = random.choice(poses_avail)
new_pose = {}
score=0
valid_time = 0

last_time = datetime.now()
cool_down = datetime.now()

# Run the game loop
running = True
while running:
    renderer.update()
    camframe = cam.frame
    body.process_frame(camframe)
    renderer.drawPose(body.points, (0,0,0), 5)
    y,x,_ = renderer.get_frame().shape
    valid = False
    if queue.getFirstFromQueue() == 0:
        new_pose = shadow.calculatePose([y/2,x/2], x,y, random.choice(poses_avail))
        print(new_pose)
        queue.addToQueue(1)
        queue.forwardQueue()
    elif queue.getFirstFromQueue() == 1:
        valid = shadow.checkPose(body.points)
        now = datetime.now()
        timedelta = (now - last_time).total_seconds()
        cooldowntimedelta = (now - cool_down).total_seconds()
        if cooldowntimedelta < 3 and valid:
            queue.addToQueue(4)
            queue.forwardQueue()
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
        cool_down = now
    elif queue.getFirstFromQueue() == 2:
        pose_number = random.choice(poses_avail)
        queue.addToQueue(0)
        queue.forwardQueue()
    elif queue.getFirstFromQueue() == 3:
        pass
    elif queue.getFirstFromQueue() == 4:
        pass


    if queue.getFirstFromQueue() != 4:
        color = (0,255,255-valid_time) if valid else (0,128,255-valid_time)
        renderer.drawPose(new_pose,color,20)


    cv2.imshow("\"Game\"", renderer.get_frame())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(shadow.get_angles(body.points, deg=True))
        break

cam.close()
cv2.destroyAllWindows()
