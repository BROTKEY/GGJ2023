import yaml
import numpy as np
import mediapipe as mp
from PIL import Image
import cv2
import math

angle_tol = np.pi/5
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

    def __init__(self, xy_size, background, camera_size):
        self.bg = cv2.resize(background, xy_size)
        self.ratio = xy_size[1] / camera_size[0]
        self.shape = np.array((camera_size[1]*self.ratio, camera_size[0]*self.ratio)).astype(int)
        self.pose_offset = np.array(((xy_size[0] - camera_size[1]) / 2, 0))
        self.update()

    def update(self):
        self.frame = self.bg.copy()

    def drawLine(self, xy1:tuple, xy2:tuple, color:tuple, thicc:int):
        self.frame = cv2.line(self.frame, xy1, xy2, color, thicc)

    def drawPose(self, points, color:tuple, thicc:int):
        thiccc = thicc+10
        shape = np.array((self.frame.shape[1], self.frame.shape[0]))
        for point in points:
            if point not in self.connections:
               continue
            for endpoint in self.connections[point]:
                if endpoint in points:
                    start_point = points[point] * self.shape + self.pose_offset/2
                    end_point = points[endpoint] * self.shape + self.pose_offset/2
                    self.frame = cv2.line(self.frame, start_point.astype(int), end_point.astype(int), color, thiccc if endpoint in range(25,29) or (point==23 and endpoint==24) else thicc)
        if 0 in points:
            point = points[0] * self.shape + self.pose_offset/2
            magnitude = np.linalg.norm(points[11] - points[12])
            self.frame = cv2.circle(self.frame, point.astype(int), int(magnitude*300), color, thicc)

    def drawImage(self, image, xy, size):
        img = cv2.resize(image, size)
        img = Image.fromarray(img)
        frame = Image.fromarray(self.frame)
        frame = frame.convert("RGBA")
        mask= img if img.format == "RGBA" else None
        frame.paste(img, xy, mask=mask)
        self.frame = np.array(frame.convert("RGB"))

    def drawText(self, text, xy, size):
        self.frame = cv2.putText(self.frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, (0,0,0), 2, cv2.LINE_AA)

    def get_frame(self):
        return self.frame


class Camera():
    def __init__(self, device=0):
        self.cap = cv2.VideoCapture(0)
        _, frame = self.cap.read()
        self.shapevar = frame.shape
    @property
    def frame(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        return frame

    @property
    def shape(self):
        return self.shapevar

    def close(self):
        self.cap.release()

class PosesEngine():
    def __init__(self):
        self.conf = (ConfigLoader()).conf

    def calculateBodyPos(self, angle, center, point):
        return (np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]]) @ (point - center)) + center

    def calculateBody(self, center, shoulder_dist, upper_body_dist, hip_dist, angle):
        left_shoulder = self.calculateBodyPos(angle, center, center + np.array([shoulder_dist/2, upper_body_dist/2]))
        left_hip = self.calculateBodyPos(angle, center, center + np.array([hip_dist/2, -upper_body_dist/2]))
        right_shoulder = self.calculateBodyPos(angle, center, center + np.array([-shoulder_dist/2, upper_body_dist/2]))
        right_hip = self.calculateBodyPos(angle, center, center + np.array([-hip_dist/2, -upper_body_dist/2]))
        return (left_hip, left_shoulder, right_hip, right_shoulder)

    def calculatePartFromAngle(self, origin, shoulder_dist, ratio, angle):
        return [origin[0] + np.sin(angle) * (shoulder_dist*ratio), origin[1] + np.cos(angle)*(shoulder_dist*ratio)]

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

    def get_angles(self, landmarks, deg=False):
        d = dict()
        for key in self.body_bones:
            bone = self.body_bones[key]
            if key == "body":
                start = (landmarks[bone[0]] + landmarks[bone[2]])/2
                stop = (landmarks[bone[1]] + landmarks[bone[3]])/2
            else:
                start = landmarks[bone[0]]
                stop =  landmarks[bone[1]]
            dr = stop - start
            d[key] = np.arctan2(dr[0], dr[1])
        if deg:
            d = {k : d[k]*180/np.pi for k in d}
        return d

    def checkPose(self, landmarks):
        if not landmarks:
            return False
        magnitude = np.linalg.norm(landmarks[11] - landmarks[12])
        # if self.last_shoulder_dist - 0.1 <= magnitude <= self.last_shoulder_dist + 0.1:
        if True:
            angles = self.get_angles(landmarks)

            body = []
            for key in self.body_bones["body"]:
                body.append(landmarks[key])
            center = np.mean(np.array(body), 0)

            # if np.linalg.norm(center - self.last_center) > 0.1:
            #     return False

            valid = True
            testing_pose = self.conf[self.last_pose_number].items()
            for key, value in testing_pose:
                if key == "shoulder_dist":
                    continue
                if not valid:
                    return valid
                anglediff = (angles[key] - value + math.pi + math.pi*2) % (math.pi*2) - math.pi
                valid = (anglediff <= angle_tol and anglediff >= -angle_tol)
                valid = valid

            return valid
        return False

    def invertAngle(self, angle):
        return (angle + math.pi) % (2 * math.pi)


    def calculatePose(self, center, screenSizeX,screenSizeY, pose_number):
        center = np.array(center)/ np.array([screenSizeY, screenSizeX])

        # shoulder_dist = float(self.conf[pose_number]["shoulder_dist"])
        try:
            shoulder_dist = float(self.conf[pose_number]["shoulder_dist"])
        except:
            shoulder_dist = 0.1
        upper_body_dist = shoulder_dist * 1.8
        hip_dist = shoulder_dist * 0.8

        body = self.calculateBody(center, shoulder_dist, upper_body_dist, hip_dist, float(self.conf[pose_number]["body"]))

        elbow_l = self.calculatePartFromAngle(body[1], shoulder_dist,0.5, float(self.conf[pose_number]["upper_arm_r"]))
        elbow_r = self.calculatePartFromAngle(body[3], shoulder_dist,0.5, float(self.conf[pose_number]["upper_arm_l"]))
        wrist_r = self.calculatePartFromAngle(elbow_r, shoulder_dist,1, float(self.conf[pose_number]["lower_arm_l"]))
        wrist_l = self.calculatePartFromAngle(elbow_l, shoulder_dist,1, float(self.conf[pose_number]["lower_arm_r"]))

        knee_l = self.calculatePartFromAngle(body[0], shoulder_dist,1, float(self.conf[pose_number]["upper_leg_r"]))
        knee_r = self.calculatePartFromAngle(body[2], shoulder_dist,1, float(self.conf[pose_number]["upper_leg_l"]))
        ankle_l = self.calculatePartFromAngle(knee_l, shoulder_dist,1, float(self.conf[pose_number]["lower_leg_r"]))
        ankle_r = self.calculatePartFromAngle(knee_r, shoulder_dist,1, float(self.conf[pose_number]["lower_leg_l"]))

        self.last_shoulder_dist = shoulder_dist
        self.last_pose_number = pose_number
        self.last_center = center


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

class ActionQueue:
    def __init__(self):
        self.actions ={0: "new_pose", 1: "check_valid", 2: "gen_new_number", 3: "wait_for_event"}
        self.queue = [0]

    def addToQueue(self,action):
        if action in self.actions.keys():
            self.queue.append(action)

    def getFirstFromQueue(self):
        return self.queue[0]

    def getQueue(self):
        return self.queue

    def forwardQueue(self):
        self.queue.pop(0)
