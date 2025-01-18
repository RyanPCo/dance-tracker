import cv2

video_path = "./boywithluv_cut.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    cv2.imshow("Video", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("Video playback stopped.")
        break

cap.release()
cv2.destroyAllWindows()


