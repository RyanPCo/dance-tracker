from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import json
import collections

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")  # Use YOLOv8 Nano model for faster inference

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Parameters for cropping, resizing, and smoothing
RESIZE_DIM = (256, 256)  # Resize cropped regions for MediaPipe Pose
SMOOTHING_WINDOW = 5  # Smoothing window size

# History for landmark smoothing
landmark_history = collections.defaultdict(list)

# Path to your video
video_path = "boywithluv_real.mov"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Prepare to save keypoints for multiple people
keypoints_data = []

# Function to smooth landmarks
def smooth_landmarks(landmarks, person_id):
    num_landmarks = 33  # MediaPipe Pose has 33 keypoints
    # Pad or truncate landmarks to ensure consistent length
    if len(landmarks) < num_landmarks:
        landmarks += [{"x": 0, "y": 0, "z": 0, "visibility": 0}] * (num_landmarks - len(landmarks))
    elif len(landmarks) > num_landmarks:
        landmarks = landmarks[:num_landmarks]

    if person_id not in landmark_history:
        landmark_history[person_id] = [landmarks]
    else:
        landmark_history[person_id].append(landmarks)
        if len(landmark_history[person_id]) > SMOOTHING_WINDOW:
            landmark_history[person_id].pop(0)

    # Calculate average position
    smoothed_landmarks = []
    for i in range(num_landmarks):
        avg_x = np.mean([frame[i]["x"] for frame in landmark_history[person_id]])
        avg_y = np.mean([frame[i]["y"] for frame in landmark_history[person_id]])
        avg_z = np.mean([frame[i]["z"] for frame in landmark_history[person_id]])
        avg_visibility = np.mean([frame[i]["visibility"] for frame in landmark_history[person_id]])
        smoothed_landmarks.append({"x": avg_x, "y": avg_y, "z": avg_z, "visibility": avg_visibility})

    return smoothed_landmarks

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    h, w, _ = frame.shape  # Get frame dimensions

    # Run YOLO detection on the frame
    results = model(frame, classes=[0])  # Detect only class 0 (person)
    detections = results[0].boxes.xyxy  # Bounding boxes [x1, y1, x2, y2]

    frame_data = {"frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)), "people": []}

    for person_id, bbox in enumerate(detections):
        x1, y1, x2, y2 = map(int, bbox)

        # Crop and resize the person's region
        person_frame = frame[y1:y2, x1:x2]
        person_rgb = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
        person_rgb_resized = cv2.resize(person_rgb, RESIZE_DIM)

        # Process the cropped person with MediaPipe Pose
        results = pose.process(person_rgb_resized)

        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                if lm.visibility > 0.5:  # Filter by visibility
                    landmarks.append({
                        "x": x1 + int(lm.x * (x2 - x1)),
                        "y": y1 + int(lm.y * (y2 - y1)),
                        "z": lm.z,
                        "visibility": lm.visibility
                    })

            # Smooth landmarks
            smoothed_landmarks = smooth_landmarks(landmarks, person_id)

            # Save landmarks for the detected person
            frame_data["people"].append({"landmarks": smoothed_landmarks})

            # Draw landmarks on the original frame for visualization
            mp_drawing.draw_landmarks(
                frame[y1:y2, x1:x2], results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

    keypoints_data.append(frame_data)

    # Display the frame with YOLO bounding boxes and MediaPipe landmarks
    for bbox in detections:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
    cv2.imshow("YOLO + MediaPipe Pose (Fixed)", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save keypoints to a JSON file
output_path = "multi_person_keypoints.json"
with open(output_path, "w") as f:
    json.dump(keypoints_data, f)

print(f"Keypoints saved to {output_path}")
