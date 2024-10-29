import mediapipe as mp
import numpy as np
import cv2

if __name__ == '__main__':

    poseMesh = mp.solutions.pose
    pose = poseMesh.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)
    draw = mp.solutions.drawing_utils
    drawStyle = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            poseMesh.POSE_CONNECTIONS,
            landmark_drawing_spec=drawStyle.get_default_pose_landmarks_style())

        cv2.imshow('Mediapipe Feed', frame)

        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
