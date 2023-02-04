import yaml
import numpy as np
import mediapipe as mp

# class GameEngine():

class BodyEngine():
    def __init__(self):
        self.landmarks = yaml.load(open("landmarks.yaml", "r"), Loader=yaml.FullLoader)
        self.landmark_names = {v: k for k, v in self.landmarks.items()}
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
        self.points = {}
        for i in range(33):
            self.points.setdefault(i, np.array((0,0)))

    def process_frame(self, frame):
        result = self.pose.process(frame)
        for i in result.pose_landmarks.landmark:
            self.points[i] = np.array((i.x, i.y))
    
    @property
    def points(self):
        return self.points
    
    def get_point_from_name(self, name):
        return self.points[self.landmark_names[name]]

class ShadowEngine():
    def __init__(self, frame):
        self.frame = frame

    def loadConfig(self,path):
        self.poses = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        print(self.poses)

    def drawPose(self, landmarks):
        pass

if __name__ == "__main__":
    x = ShadowEngine()
    x.loadConfig("poses.yaml")