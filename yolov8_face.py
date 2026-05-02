import cv2
import torch
from ultralytics import YOLO
from inference import Inferencer  # Import the Inferencer class

# Load your custom YOLOv8 model for face detection
face_model = YOLO('yolov8n-face.pt')

# Load your emotion classification model
emotion_model_path = "checkpoints/8.pt"  # Adjust this to your saved model path
emotion_inferencer = Inferencer(emotion_model_path)

# Define emotion labels
emotion_labels = ['Anger', 'Happy','Surprise','Sad', 'Contempt', 'Fear',  'Disgust', 'Neutral', ]  # Adjust these labels based on your model's classes

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame for face detection
    results = face_model(frame)

    # Extract bounding boxes, classify emotions, and plot them
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Extract face region
            face_img = frame[y1:y2, x1:x2]
            
            # Classify emotion
            emotion_class, emotion_confidence = emotion_inferencer.predict(face_img)
            emotion_label = emotion_labels[emotion_class]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put emotion label on the bounding box
            label = f"{emotion_label}: {emotion_confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with bounding boxes and emotion labels
    cv2.imshow("Face Detection and Emotion Classification", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()