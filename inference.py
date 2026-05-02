import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2

# ============================================================
# CS731 Emotion Recognition - Inference Helper
# Loads a trained model checkpoint and predicts emotion
# from a face image crop (used by yolov8_face.py and GUI)
# ============================================================

# 8 emotion classes — order MUST match dataset folder order
# Dataset loader uses os.listdir() which is alphabetical
EMOTION_LABELS = [
    'anger',      # index 0
    'contempt',   # index 1
    'disgust',    # index 2
    'fear',       # index 3
    'happy',      # index 4
    'neutral',    # index 5
    'sad',        # index 6
    'surprise'    # index 7
]

# Display names with capital letters for GUI
EMOTION_DISPLAY = {
    'anger':    'Angry',
    'contempt': 'Contempt',
    'disgust':  'Disgust',
    'fear':     'Fear',
    'happy':    'Happy',
    'neutral':  'Neutral',
    'sad':      'Sad',
    'surprise': 'Surprise'
}

# Emoji for each emotion — used in chatbot responses
EMOTION_EMOJI = {
    'anger':    '😠',
    'contempt': '😒',
    'disgust':  '🤢',
    'fear':     '😨',
    'happy':    '😊',
    'neutral':  '😐',
    'sad':      '😢',
    'surprise': '😲'
}


class Inferencer:
    def __init__(self, model_path):
        """
        Load a trained emotion model checkpoint and prepare for inference.

        Args:
            model_path (str): Path to the saved .pt checkpoint file
        """
        # Use GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Inferencer using device: {self.device}")

        # Load the model from checkpoint
        self.model = self.load_model(model_path)

        # IMPORTANT: These transforms MUST match the test transforms in train.py
        # Using ImageNet normalization values (not 0.5) to match our improved training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),          # Resize face crop to model input size
            transforms.ToTensor(),                  # Convert PIL image to tensor [0, 1]
            transforms.Normalize(                   # Normalize using ImageNet mean/std
                mean=[0.485, 0.456, 0.406],         # Same as train.py test_transform
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_model(self, model_path):
        """
        Load entire model from .pt file.

        Args:
            model_path (str): Path to checkpoint

        Returns:
            model: Loaded PyTorch model in eval mode
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        # weights_only=False loads the full model object (not just weights)
        model = torch.load(model_path, map_location=self.device, weights_only=False)
        model.eval()   # Set to evaluation mode — disables dropout, freezes batch norm
        print(f"Model loaded from: {model_path}")
        return model

    def preprocess_image(self, image):
        """
        Preprocess a raw webcam face crop for model input.

        Args:
            image: OpenCV BGR image (numpy array) — face crop from YOLOv8

        Returns:
            tensor: Preprocessed image tensor ready for model
        """
        # OpenCV reads images in BGR — convert to RGB for PIL/PyTorch
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

        # Apply transforms (resize, normalize)
        image = self.transform(image)

        # Add batch dimension: [C, H, W] → [1, C, H, W]
        return image.unsqueeze(0).to(self.device)

    def predict(self, face_image):
        """
        Predict emotion from a face image crop.

        Args:
            face_image: OpenCV BGR image (numpy array) — face crop

        Returns:
            tuple: (emotion_label, display_name, confidence, emoji)
                   e.g. ('happy', 'Happy', 0.92, '😊')
        """
        # Preprocess the face crop
        image_tensor = self.preprocess_image(face_image)

        # Run inference without computing gradients (saves memory)
        with torch.no_grad():
            output = self.model(image_tensor)   # Raw logits from model

            # Convert logits to probabilities with softmax
            probabilities = torch.nn.functional.softmax(output, dim=1)

            # Get highest probability class index
            predicted_idx = torch.argmax(probabilities, dim=1).item()

            # Get confidence score for predicted class
            confidence = probabilities[0][predicted_idx].item()

        # Convert index to emotion label
        emotion_label = EMOTION_LABELS[predicted_idx]
        display_name  = EMOTION_DISPLAY[emotion_label]
        emoji         = EMOTION_EMOJI[emotion_label]

        return emotion_label, display_name, confidence, emoji


# -----------------------------------------------------------
# Quick test — run directly to verify model works
# -----------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Use best.pt from EfficientNet (our best model)
    model_path = "checkpoints/efficientnet/best.pt"

    if not os.path.exists(model_path):
        print(f"Checkpoint not found: {model_path}")
        print("Available checkpoints:")
        for root, dirs, files in os.walk("checkpoints"):
            for f in files:
                if f.endswith('.pt'):
                    print(f"  {os.path.join(root, f)}")
        sys.exit(1)

    print("Loading inferencer...")
    inferencer = Inferencer(model_path)

    # Test with a sample image from test_images
    test_image_path = "test_images/happy"
    if os.path.exists(test_image_path):
        images = os.listdir(test_image_path)
        if images:
            img_path = os.path.join(test_image_path, images[0])
            image = cv2.imread(img_path)
            if image is not None:
                label, display, confidence, emoji = inferencer.predict(image)
                print(f"Test prediction: {emoji} {display} ({confidence:.1%} confidence)")
