from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video_path = "./boywithluv_real.mov"
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Run YOLO detection on the current frame
    results = model(frame)

    # Draw bounding boxes for detected people
    for detection in results[0].boxes.xyxy:  # Bounding boxes [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, detection)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    # Display the frame with bounding boxes
    cv2.imshow("YOLOv8 Person Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
