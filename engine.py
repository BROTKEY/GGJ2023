import yaml
import numpy as np
import mediapipe as mp
from PIL import Image
import cv2
import math

class GameEngine():
    connections = {
        11 : [12,13,23],
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

    def __init__(self, cap):
        self.cap = cap
        self.update()

    def update(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        self.frame = frame

    def drawPose(self, points, color:tuple, thicc:int):
        shape = np.array((self.frame.shape[1], self.frame.shape[0]))
        for point in points:
            if point not in self.connections:
               continue
            for endpoint in self.connections[point]:
                if endpoint in points:
                    start_point = points[point] * shape
                    end_point = points[endpoint] * shape
                    self.frame = cv2.line(self.frame, start_point.astype(int), end_point.astype(int), color, thicc)

    def drawImage(self, image, xy, size):
        img = cv2.resize(image, size)
        img = Image.fromarray(img)
        frame = Image.fromarray(self.frame)
        frame = frame.convert("RGBA")
        frame.paste(img, xy, mask=img)
        self.frame = np.array(frame.convert("RGB"))

    def get_frame(self):
        return self.frame

    def close(self):
        self.cap.release()

class PosesEngine():
    def __init__(self):
        self.conf = (ConfigLoader()).conf

    def calculateBodyPos(self, angle, center, point):
        return (np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]]) @ (point - center)) + center
        
    def calculateBody(self, center, shoulder_dist, upper_body_dist, hip_dist, angle):
        left_shoulder = self.calculateBodyPos(angle, center, center + np.array([shoulder_dist/2, upper_body_dist/2]))
        left_hip = self.calculateBodyPos(angle, center, center + np.array([hip_dist/2, -upper_body_dist/2]))
        right_shoulder = self.calculateBodyPos(angle, center, center + np.array([-shoulder_dist/2, upper_body_dist/2]))
        right_hip = self.calculateBodyPos(angle, center, center + np.array([-hip_dist/2, -upper_body_dist/2]))
        return (left_hip, left_shoulder, right_hip, right_shoulder)

    def calculatePartFromAngle(self, origin, shoulder_dist, ratio, angle):
        return origin + [np.cos(angle)*(shoulder_dist*ratio), np.sin(angle) * shoulder_dist]

    def toCoords(self, body, elbow_l, elbow_r, wrist_l, wrist_r, knee_l, knee_r, ankle_l, ankle_r):
        return {
            23: body[0],
            11: body[1],
            24: body[2],
            12: body[3],
            13: elbow_l,
            15: wrist_l,
            14: elbow_r,
            16: wrist_r,
            25: knee_l,
            26: knee_r,
            27: ankle_l,
            28: ankle_r,
        }
    
    def calculatePose(self, center, screenSizeX,screenSizeY, number):
        center = np.array(center)/ np.array([screenSizeY, screenSizeX])
        print(center)

        
        shoulder_dist = float(self.conf[number]["shoulder_dist"])
        upper_body_dist = shoulder_dist * 1.8
        hip_dist = shoulder_dist * 0.8

        body = self.calculateBody(center, shoulder_dist, upper_body_dist, hip_dist, float(self.conf[number]["body"]))
        
        elbow_l = self.calculatePartFromAngle(body[1], shoulder_dist,0.6, float(self.conf[number]["upper_arm_l"]))
        elbow_r = self.calculatePartFromAngle(body[3], shoulder_dist,0.6, float(self.conf[number]["upper_arm_r"]))
        wrist_l = self.calculatePartFromAngle(elbow_l, shoulder_dist,1, float(self.conf[number]["lower_arm_l"]))
        wrist_r = self.calculatePartFromAngle(elbow_r, shoulder_dist,1, float(self.conf[number]["lower_arm_r"]))

        knee_l = self.calculatePartFromAngle(body[0], shoulder_dist,1, float(self.conf[number]["upper_leg_l"]))
        knee_r = self.calculatePartFromAngle(body[2], shoulder_dist,1, float(self.conf[number]["upper_leg_r"]))
        ankle_l = self.calculatePartFromAngle(knee_l, shoulder_dist,1, float(self.conf[number]["lower_leg_l"]))
        ankle_r = self.calculatePartFromAngle(knee_r, shoulder_dist,1, float(self.conf[number]["lower_leg_r"]))

        return self.toCoords(body, elbow_l, elbow_r, wrist_l, wrist_r, knee_l, knee_r, ankle_l, ankle_r)

class BodyEngine():
    def __init__(self):
        self.landmarks = yaml.load(open("landmarks.yaml", "r"), Loader=yaml.FullLoader)
        self.landmark_names = {v: k for k, v in self.landmarks.items()}
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
        self.marks = {}

    def process_frame(self, frame):
        result = self.pose.process(frame)
        if not result.pose_landmarks:
            return
        for n, i in enumerate(result.pose_landmarks.landmark):
            self.marks[n] = np.array((i.x, i.y))

    @property
    def points(self):
        return self.marks

    def get_point_from_name(self, name):
        return self.marks[self.landmark_names[name]]
    
class ConfigLoader:
    def __init__(self):
        raw = yaml.safe_load(open("./poses.yaml","r"))
        for key in raw:
            for parts in raw[key]:
                if parts == "shoulder_dist":
                    continue
                raw[key][parts] = math.radians(raw[key][parts])
        self.conf = raw
