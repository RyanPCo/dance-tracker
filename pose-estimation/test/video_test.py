import cv2
import mediapipe as mp
import json

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

model_path = "./models/pose_landmarker_heavy.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_poses=3,
    min_pose_detection_confidence=0.1,
    min_pose_presence_confidence=0.1,
    min_tracking_confidence=0.1,
    running_mode=VisionRunningMode.VIDEO,
)

pose_landmarker = PoseLandmarker.create_from_options(options)

video_path = "./videos/boywithluv_real.mov"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

keypoints_data = []

fps = cap.get(cv2.CAP_PROP_FPS)
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe Image object
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Use the current frame's timestamp in ms (approximate)
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms == 0 and frame_index == 0:
        # If CAP_PROP_POS_MSEC is 0 at the very start, it's likely just the first frame
        timestamp_ms = 0
    elif timestamp_ms == 0:
        # If CAP_PROP_POS_MSEC fails, estimate from the frame index
        timestamp_ms = int(frame_index * (1000 / fps))

    # Run pose detection in VIDEO mode
    results = pose_landmarker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)

    frame_data = {
        "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
        "people": []
    }

    # Process each detected person
    if results.pose_landmarks:
        print(f"Number of detected poses: {len(results.pose_landmarks)}")

        for person_landmarks in results.pose_landmarks:
            # Convert the list of landmarks to NormalizedLandmarkList
            landmark_list = landmark_pb2.NormalizedLandmarkList()
            for lm in person_landmarks:
                landmark = landmark_list.landmark.add()
                landmark.x = lm.x
                landmark.y = lm.y
                landmark.z = lm.z
                landmark.visibility = lm.visibility

            # Draw landmarks for each person individually
            mp_drawing.draw_landmarks(
                frame, 
                landmark_list,  # Use the converted NormalizedLandmarkList
                mp_pose.POSE_CONNECTIONS
            )

            # Prepare landmarks data for JSON
            landmarks = [
                {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                } 
                for lm in person_landmarks
            ]
            frame_data["people"].append({"landmarks": landmarks})

    keypoints_data.append(frame_data)

    # Display the frame with overlaid landmarks
    cv2.imshow("MediaPipe Multi-Person Pose", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# Save keypoints to a JSON file
output_path = "multi_person_keypoints_mediapipe.json"
with open(output_path, "w") as f:
    json.dump(keypoints_data, f)

print(f"Keypoints saved to {output_path}")
