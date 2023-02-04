import yaml
import numpy as np
import mediapipe as mp
from PIL import Image
import cv2

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
        frame.paste(img, xy)
        self.frame = np.array(frame)

    def get_frame(self):
        return self.frame

    def close(self):
        self.cap.release()

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
