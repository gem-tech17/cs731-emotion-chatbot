from ultralytics import YOLO
import cv2
face_model = YOLO('yolov8n-face.pt') # path to checkpoint

frame = cv2.imread('image.png') #load an image
results = face_model(frame) #make prediction with model

boxes = results[0].boxes.xyxy.cpu().numpy() 
print(boxes) 
