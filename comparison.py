import json
import numpy as np

def calculate_similarity(frame1, frame2):
    distances = []
    for lm1, lm2 in zip(frame1["landmarks"], frame2["landmarks"]):
        dist = np.linalg.norm(np.array([lm1["x"], lm1["y"], lm1["z"]]) -
                              np.array([lm2["x"], lm2["y"], lm2["z"]]))
        distances.append(dist)
    return np.mean(distances)

with open("normalized_keypoints_video1.json", "r") as f1, open("normalized_keypoints_video2.json", "r") as f2:
    video1_data = json.load(f1)
    video2_data = json.load(f2)

similarity_scores = []
for frame1, frame2 in zip(video1_data, video2_data):
    score = calculate_similarity(frame1, frame2)
    similarity_scores.append(score)

average_score = np.mean(similarity_scores)
final_score = max(0, 100 - average_score * 100)
print(f"Overall Similarity Score: {final_score:.2f}/100")
