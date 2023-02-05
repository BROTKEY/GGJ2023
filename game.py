import cv2
import mediapipe as mp
import numpy as np
from random import randint
from engine import BodyEngine, GameEngine, PosesEngine, Camera
from engine import BodyEngine, GameEngine, PosesEngine, ActionQueue
from datetime import datetime
import time
import random, yaml
import simpleaudio as sa

background = cv2.imread("./img/background.png")
cam = Camera()
renderer = GameEngine((1280,720), background, cam.shape)
body = BodyEngine()
shadow = PosesEngine()
queue = ActionQueue()
success_wav = sa.WaveObject.from_wave_file("success.wav")

startscreee = False

shadow_color = (100,100,100)
shadow_thickness = 10
skeleton_color = (0,0,0)
skeleton_thickness = 4
poses_avail = list(yaml.load(open("poses.yaml","r"), Loader=yaml.FullLoader).keys())
new_pose = {}
valid_time = 0
camera_enabled = False

last_time = datetime.now()
running = True

black = cv2.imread("Black.png")

if startscreen:
    # Startscreen
    while True:
        renderer.update()
        camframe = cam.frame
        renderer.drawImage(black, (0,0), (renderer.frame.shape[1], renderer.frame.shape[0]))
        # renderer.drawImage(camframe, (int((renderer.shape[0]-cam.shape[0]*renderer.ratio)/1.5),0), (np.array((cam.shape[1], cam.shape[0]))*renderer.ratio).astype(int))
        body.process_frame(camframe)
        renderer.drawPose(body.points, (0,0,255), 2)
        y,x,_ = renderer.get_frame().shape
        target_pose = shadow.calculatePose([y/2,x/2], x,y, 0)
        valid, acc = shadow.checkPose(body.points)
        print(acc)
        now = datetime.now()
        timedelta = (now - last_time).total_seconds()
        renderer.drawText("T-Pose to start", (508,64), 1)
        if valid:
            valid_time += 200 * timedelta
            if valid_time > 255:
                valid_time = 0
                break
        else:
            if valid_time < 0:
                valid_time = 0
            else:
                valid_time -= 25 * timedelta
        last_time = now

        renderer.drawPose(target_pose, (0,255,255-valid_time), 20)
        cv2.imshow("\"Game\"", renderer.get_frame())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break


# Run the game loop
while running:
    renderer.update()
    camframe = cam.frame
    body.process_frame(camframe)
    if camera_enabled: renderer.drawImage(camframe, (int((renderer.shape[0]-cam.shape[0]*renderer.ratio)/1.5),0), (np.array((cam.shape[1], cam.shape[0]))*renderer.ratio).astype(int))
    renderer.drawPose(body.points, (0,0,0), 5)
    y,x,_ = renderer.get_frame().shape
    valid = False

    if queue.getFirstFromQueue() == 0:
        new_pose = shadow.calculatePose([3*y/4 , x/4], x,y, random.choice(poses_avail))
        print(new_pose)
        queue.addToQueue(1)
        queue.forwardQueue()
    elif queue.getFirstFromQueue() == 1:
        valid, acc = shadow.checkPose(body.points)
        print(acc)
        now = datetime.now()
        timedelta = (now - last_time).total_seconds()
        if valid:
            valid_time += 100 * timedelta
            if valid_time > 255:
                valid_time = 0
                queue.addToQueue(0)
                queue.forwardQueue()
                success_wav.play()
        else:
            if valid_time < 0:
                valid_time = 0
            else:
                valid_time -= 25 * timedelta
        last_time = now
    elif queue.getFirstFromQueue() == 3:
        pass

    color = (0,255,255-valid_time) if valid else (0,128,255-valid_time)
    renderer.drawPose(new_pose,color,10)


    cv2.imshow("\"Game\"", renderer.get_frame())
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print(shadow.get_angles(body.points, deg=True))
        break
    elif key == ord("c"):
        camera_enabled = not camera_enabled
    elif key == ord("e"):
        queue.addToQueue(0)
        queue.forwardQueue()

cam.close()
cv2.destroyAllWindows()
