import cv2
import mediapipe as mp
import json
import sys

# Import the necessary classes from MediaPipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Define the path to your MediaPipe pose landmarker model
model_path = "./models/pose_landmarker_heavy.task"  # Ensure this is the heavy model

# Initialize MediaPipe PoseLandmarker in IMAGE mode
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
ImageFormat = mp.ImageFormat  # Ensure Image and ImageFormat are correctly imported

# Set up PoseLandmarker options for IMAGE mode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_poses=4,
    min_pose_detection_confidence=0.1,
    min_pose_presence_confidence=0.1,
    min_tracking_confidence=0.1,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,  # Set to IMAGE mode
)

# Create the PoseLandmarker instance
pose_landmarker = PoseLandmarker.create_from_options(options)

# Path to your input video
video_path = "./videos/boywithluv_real.mov"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Prepare to save keypoints
keypoints_data = []

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video Properties:\n- FPS: {fps}\n- Resolution: {width}x{height}\n- Total Frames: {total_frames}")

frame_index = 0  # To keep track of the current frame number

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no frame is returned

    # Convert BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe Image object
    try:
        mp_image = vision.Image(
            image_format=ImageFormat.SRGB,
            data=rgb_frame
        )
    except AttributeError:
        # Fallback if vision.Image is not available
        mp_image = mp.Image(
            image_format=ImageFormat.SRGB,
            data=rgb_frame
        )

    # Run pose detection on the image
    results = pose_landmarker.detect(mp_image)

    # Prepare storage for this frame
    frame_data = {
        "frame": frame_index,
        "people": []
    }

    # Process each detected pose
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
            

    # Append frame data to the keypoints list
    keypoints_data.append(frame_data)

    # Display the frame with overlaid landmarks
    cv2.imshow("MediaPipe Multi-Person Pose (Image Mode)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit if 'q' is pressed

    frame_index += 1  # Increment frame index

    # Optional: Print progress every 100 frames
    if frame_index % 100 == 0:
        print(f"Processed {frame_index}/{total_frames} frames...")

cap.release()
cv2.destroyAllWindows()

# Save keypoints to a JSON file
output_path = "./data/multi_person_keypoints.json"
with open(output_path, "w") as f:
    json.dump(keypoints_data, f, indent=2)

print(f"Keypoints saved to {output_path}")
