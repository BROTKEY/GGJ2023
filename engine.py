import yaml
import numpy as np
import mediapipe as mp
import cv2

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

class ShadowEngine():
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

    def loadConfig(self,path):
        self.poses = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        print(self.poses)

    def update(self):
        _, frame = self.cap.read()
        
    def calculateBody(self, center, shoulder_dist, upper_body_dist, hip_dist):
        left_shoulder = center + np.array([shoulder_dist/2, upper_body_dist/2])
        left_hip = center + np.array([shoulder_dist/2, -upper_body_dist/2])
        right_shoulder = center + np.array([-shoulder_dist/2, upper_body_dist/2])
        right_hip = center + np.array([-shoulder_dist/2, -upper_body_dist/2])
        return (left_hip, left_shoulder,right_hip, right_shoulder )

    def calculatePartFromAngle(self, origin, shoulder_dist, angle):
        return origin + [np.cos(angle)*shoulder_dist, np.sin(angle) *len]

    def toCoords(self, body, elbow_l, elbow_r, wrist_l, wrist_r, knee_l, knee_r, ankle_l, ankle_r):
        return {
            23: body[0],
            11: body[1],
            24
        }

    def drawPose(self, center, screenSizeX,screenSizeY):
        center = np.array(center)/ np.array([screenSizeX, screenSizeY])

        shoulder_dist = float(self.poses["shoulder_dist"])
        upper_body_dist = shoulder_dist * 1.5
        hip_dist = shoulder_dist * 0.8

        body = self.calculateBody(center, shoulder_dist, upper_body_dist, hip_dist)

        elbow_l = self.calculatePartFromAngle(body[1], shoulder_dist, float(self.poses["upper_arm_l"]))
        elbow_r = self.calculatePartFromAngle(body[2], shoulder_dist, float(self.poses["upper_arm_r"]))
        wrist_l = self.calculatePartFromAngle(elbow_l, shoulder_dist, float(self.poses["lower_arm_l"]))
        wrist_r = self.calculatePartFromAngle(elbow_r, shoulder_dist, float(self.poses["lower_arm_r"]))

        knee_l = self.calculatePartFromAngle(body[0], shoulder_dist, float(self.poses["upper_leg_l"]))
        knee_r = self.calculatePartFromAngle(body[3], shoulder_dist, float(self.poses["upper_leg_r"]))
        ankle_l = self.calculatePartFromAngle(knee_l, shoulder_dist, float(self.poses["lower_leg_l"]))
        ankle_r = self.calculatePartFromAngle(knee_r, shoulder_dist, float(self.poses["lower_leg_r"]))

        return (body, elbow_l, elbow_r, wrist_l, wrist_r, knee_l, knee_r, ankle_l, ankle_r)

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame = frame

    def drawPose(self, points, color:tuple, thicc:int):
        shape = np.array((self.frame.shape[0], self.frame.shape[1]))
        for point in points:
            if point not in self.connections:
               continue 
            for endpoint in self.connections[point]:
                if endpoint in points:
                    start_point = points[point] * shape
                    end_point = points[endpoint] * shape
                    self.frame = cv2.line(self.frame, start_point.astype(int), end_point.astype(int), color, thicc)

    def get_frame(self):
        return self.frame
    
    def close(self):
        self.cap.release()