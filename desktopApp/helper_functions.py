pose_silhouette = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER',
                   'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
                   'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST',
                   'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
                   'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                   'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
exceptions_silhouette = ['LEFT_EYE_INNER', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE_OUTER',
                         'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT']

exercises_dict = {
    0: [50, 5],     #all
    1: [25, 20],    #Jumping Jacks
    2: [50, 40],    #Side Leg Squat
    3: [40, 30],    #Squat
    4: [25, 20],    #Standing Sit-up
    5: [70, 60],    #Side Bend
    6: [80, 60],    #Bend
}

exercises_names = {
    0: ['Nothing'],
    1: ['Jumping Jack'],
    2: ['Side Leg Squat'],
    3: ['Squat'],
    4: ['Standing Sit-up'],
    5: ['Side Bend'],
    6: ['Bend']
}

def extract_landmarks(landmark_list):
    frame_landmarks = {}
    if landmark_list.pose_landmarks:
        for i, ps in enumerate(pose_silhouette):
            if ps in exceptions_silhouette:
                continue

            landmark_list.pose_landmarks.landmark[i].x = landmark_list.pose_landmarks.landmark[i].x * 720
            landmark_list.pose_landmarks.landmark[i].y = landmark_list.pose_landmarks.landmark[i].y * 1280
            frame_landmarks.update({ps: [landmark_list.pose_landmarks.landmark[i].x,
                                         landmark_list.pose_landmarks.landmark[i].y]})

    return frame_landmarks
