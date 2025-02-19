import json

with open("./output_colorful/alphapose-results.json", "r") as file:
    data = json.load(file)

# print(json.dumps(data, indent=4))

for pose in data:
    keypoints = pose["keypoints"]
    keypoints_xy = []
    for i in range(0, len(keypoints), 3):
        keypoints_xy.append(keypoints[i])
        keypoints_xy.append(keypoints[i+1])

print(keypoints_xy)
