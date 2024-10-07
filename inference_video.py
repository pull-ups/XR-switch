from ultralytics import YOLO
import cv2
model = YOLO('./best.pt')
video_path = './IMG_8937.mp4'
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('IMG_8937_output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)
cap.release()
out.release()
