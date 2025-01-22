import json
import numpy as np

input_path = "./keypoints.json"
with open(input_path, "r") as f:
    keypoints_data = json.load(f)

def normalize_landmarks(landmarks):
    left_hip = np.array([landmarks[23]["x"], landmarks[23]["y"], landmarks[23]["z"]])
    right_hip = np.array([landmarks[24]["x"], landmarks[24]["y"], landmarks[24]["z"]])
    hip_center = (left_hip + right_hip) / 2

    left_shoulder = np.array([landmarks[11]["x"], landmarks[11]["y"], landmarks[11]["z"]])
    right_shoulder = np.array([landmarks[12]["x"], landmarks[12]["y"], landmarks[12]["z"]])
    shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)

    normalized_landmarks = []
    for lm in landmarks:
        normalized_landmarks.append({
            "x": (lm["x"] - hip_center[0]) / shoulder_distance,
            "y": (lm["y"] - hip_center[1]) / shoulder_distance,
            "z": (lm["z"] - hip_center[2]) / shoulder_distance,
        })
    return normalized_landmarks

normalized_data = []
for frame in keypoints_data:
    normalized_landmarks = normalize_landmarks(frame["landmarks"])
    normalized_data.append({
        "frame": frame["frame"],
        "landmarks": normalized_landmarks
    })

output_path = "normalized_keypoints.json"
with open(output_path, "w") as f:
    json.dump(normalized_data, f)

print(f"Normalized keypoints saved to {output_path}")
