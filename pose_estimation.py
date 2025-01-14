import cv2
import mediapipe as mp
import json

video_path = "./boywithluv_cut.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

keypoints_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        frame_keypoints = {
            "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "landmarks": [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in results.pose_landmarks.landmark
            ]
        }
        keypoints_data.append(frame_keypoints)

    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("Video playback stopped.")
        break

cap.release()
cv2.destroyAllWindows()

output_path = "keypoints.json"
with open(output_path, "w") as f:
    json.dump(keypoints_data, f)

print(f"Keypoints saved to {output_path}")
