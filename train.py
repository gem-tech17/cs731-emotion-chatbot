import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import timm
from dataset import CustomImageDataset

# ============================================================
# CS731 Emotion Recognition - Improved Training Script
# Models supported: ConvNeXtV2 Pico, ResNet-50, EfficientNet-B0
# Change MODEL_NAME below to switch between models
# ============================================================

# -----------------------------------------------------------
# CONFIGURATION — Change these settings to experiment
# -----------------------------------------------------------

# Choose your model:
#   'convnextv2'   → ConvNeXtV2 Pico  (lecturer baseline)
#   'resnet50'     → ResNet-50
#   'efficientnet' → EfficientNet-B0  (PPT choice)
MODEL_NAME = 'efficientnet'

BATCH_SIZE = 32          # Increased from 10 — faster training on RTX 3050
LEARNING_RATE = 3e-4     # Increased from 1e-5 — converges faster
NUM_EPOCHS = 30          # Max epochs — early stopping will stop earlier if needed
PATIENCE = 5             # Stop training if accuracy doesn't improve for 5 epochs
NUM_CLASSES = 8          # 8 emotion classes (Ekman theory)

# Dataset paths
TRAIN_DIR = './train_images'
TEST_DIR  = './test_images'

# Where to save checkpoints
SAVE_PATH = f'checkpoints/{MODEL_NAME}'

# -----------------------------------------------------------
# MODEL DEFINITIONS
# -----------------------------------------------------------

def build_model(model_name, num_classes):
    """
    Build and return the selected model with modified final layer.
    All models use pretrained ImageNet weights — we only retrain
    the final classification head for our 8 emotion classes.
    """
    if model_name == 'convnextv2':
        # ConvNeXtV2 Pico — lecturer's baseline model
        model = timm.create_model('timm/convnextv2_pico.fcmae_ft_in1k', pretrained=True)
        model.head.fc = nn.Linear(512, num_classes)
        print("Model: ConvNeXtV2 Pico loaded ✓")

    elif model_name == 'resnet50':
        # ResNet-50 — classic CNN, strong baseline
        model = timm.create_model('resnet50', pretrained=True)
        in_features = model.fc.in_features          # Get original output size (2048)
        model.fc = nn.Linear(in_features, num_classes)
        print("Model: ResNet-50 loaded ✓")

    elif model_name == 'efficientnet':
        # EfficientNet-B0 — lightweight but powerful (our PPT choice)
        model = timm.create_model('efficientnet_b0', pretrained=True)
        in_features = model.classifier.in_features  # Get original output size (1280)
        model.classifier = nn.Linear(in_features, num_classes)
        print("Model: EfficientNet-B0 loaded ✓")

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose convnextv2, resnet50, or efficientnet.")

    return model

# -----------------------------------------------------------
# TRAINER CLASS
# -----------------------------------------------------------

class Trainer:
    def __init__(self, model_name, save_path):
        """
        Initialize Trainer with augmented transforms, datasets, model, and optimizer.

        Args:
            model_name (str): Which model to train ('convnextv2', 'resnet50', 'efficientnet')
            save_path (str): Directory to save model checkpoints
        """

        # ---------------------------------------------------
        # TRAINING TRANSFORMS — with data augmentation
        # Augmentation adds random variations so the model
        # learns to generalise, not just memorise the data
        # ---------------------------------------------------
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),              # Resize to standard input size
            transforms.RandomHorizontalFlip(p=0.5),     # Randomly flip faces horizontally
            transforms.RandomRotation(degrees=15),      # Rotate up to ±15 degrees
            transforms.ColorJitter(                     # Randomly change brightness/contrast
                brightness=0.3,
                contrast=0.3,
                saturation=0.2
            ),
            transforms.RandomGrayscale(p=0.05),         # Occasionally convert to grayscale
            transforms.ToTensor(),                      # Convert PIL image to tensor [0,1]
            transforms.Normalize(                       # Normalize using ImageNet mean/std
                mean=[0.485, 0.456, 0.406],             # These values match pretrained models
                std=[0.229, 0.224, 0.225]
            )
        ])

        # ---------------------------------------------------
        # TEST TRANSFORMS — no augmentation, just clean resize
        # We never augment test data — we want fair evaluation
        # ---------------------------------------------------
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.save_path = save_path
        self.model_name = model_name

        # ---------------------------------------------------
        # DATASETS AND DATALOADERS
        # ---------------------------------------------------
        print("\nLoading datasets...")
        self.train_dataset = CustomImageDataset(root_dir=TRAIN_DIR, transform=self.train_transform)
        self.test_dataset  = CustomImageDataset(root_dir=TEST_DIR,  transform=self.test_transform)

        # num_workers=2 loads data in parallel — speeds up training
        self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
        self.test_loader  = DataLoader(self.test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        print(f"Train samples : {len(self.train_dataset)}")
        print(f"Test  samples : {len(self.test_dataset)}")

        # ---------------------------------------------------
        # MODEL
        # ---------------------------------------------------
        print(f"\nBuilding model: {model_name}")
        self.model = build_model(model_name, NUM_CLASSES)

        # ---------------------------------------------------
        # LOSS FUNCTION
        # CrossEntropyLoss is standard for multi-class classification
        # It expects raw logits (before softmax) as input
        # ---------------------------------------------------
        self.criterion = nn.CrossEntropyLoss()

        # ---------------------------------------------------
        # OPTIMIZER
        # Adam with higher lr — we use pretrained weights so
        # we can afford a bigger learning rate than 1e-5
        # ---------------------------------------------------
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # ---------------------------------------------------
        # LEARNING RATE SCHEDULER
        # Reduces learning rate when accuracy stops improving
        # This helps fine-tune in the later epochs
        # ---------------------------------------------------
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )

        # ---------------------------------------------------
        # DEVICE — use GPU if available
        # ---------------------------------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.model.to(self.device)

        # Track best accuracy for early stopping and checkpoint saving
        self.best_accuracy = 0.0
        self.epochs_no_improve = 0

    # -----------------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------------
    def train(self, num_epochs):
        """
        Train the model for up to num_epochs, with early stopping.

        Args:
            num_epochs (int): Maximum number of epochs to train
        """
        print(f"\n{'='*60}")
        print(f"  Starting training: {self.model_name.upper()}")
        print(f"  Epochs: {num_epochs}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
        print(f"{'='*60}\n")

        os.makedirs(self.save_path, exist_ok=True)

        # Log file to save results for comparison later
        log_file = os.path.join(self.save_path, 'training_log.txt')

        for epoch in range(num_epochs):

            # -------------------------------------------
            # TRAINING PHASE
            # -------------------------------------------
            self.model.train()   # Enable dropout, batch norm updates
            running_loss = 0.0

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()          # Clear previous gradients

                outputs = self.model(images)         # Forward pass

                # labels are one-hot — convert to class indices for CrossEntropyLoss
                label_indices = torch.argmax(labels, dim=1)
                loss = self.criterion(outputs, label_indices)

                loss.backward()                     # Backward pass
                self.optimizer.step()               # Update weights

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"\nEpoch {epoch+1} | Loss: {avg_loss:.4f}")

            # -------------------------------------------
            # EVALUATION PHASE
            # -------------------------------------------
            accuracy = self.evaluate()

            # Adjust learning rate based on accuracy
            self.scheduler.step(accuracy)

            # -------------------------------------------
            # CHECKPOINT SAVING — save only best model
            # -------------------------------------------
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.epochs_no_improve = 0

                # Save the best checkpoint
                best_path = os.path.join(self.save_path, 'best.pt')
                torch.save(self.model, best_path)
                print(f"  ✅ New best accuracy: {accuracy:.2f}% — checkpoint saved!")

            else:
                self.epochs_no_improve += 1
                print(f"  No improvement for {self.epochs_no_improve}/{PATIENCE} epochs")

            # Also save every epoch checkpoint (epoch number as filename, like original)
            epoch_path = os.path.join(self.save_path, f'{epoch}.pt')
            torch.save(self.model, epoch_path)

            # Write to log file
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%\n")

            # -------------------------------------------
            # EARLY STOPPING
            # Stop if no improvement for PATIENCE epochs
            # -------------------------------------------
            if self.epochs_no_improve >= PATIENCE:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs.")
                print(f"   Best accuracy achieved: {self.best_accuracy:.2f}%")
                break

        print(f"\n{'='*60}")
        print(f"  Training complete!")
        print(f"  Model: {self.model_name.upper()}")
        print(f"  Best Test Accuracy: {self.best_accuracy:.2f}%")
        print(f"  Best checkpoint: {self.save_path}/best.pt")
        print(f"{'='*60}\n")

    # -----------------------------------------------------------
    # EVALUATION LOOP
    # -----------------------------------------------------------
    def evaluate(self):
        """
        Evaluate the model on the test set.

        Returns:
            float: Accuracy percentage on test set
        """
        self.model.eval()   # Disable dropout, freeze batch norm
        correct = 0
        total = 0

        with torch.no_grad():   # No gradients needed during evaluation
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                predicted      = torch.argmax(outputs, dim=1)       # Model's predicted class
                label_indices  = torch.argmax(labels,  dim=1)       # True class from one-hot

                total   += labels.size(0)
                correct += (predicted == label_indices).sum().item()

        accuracy = (correct / total) * 100
        print(f"  Test Accuracy: {accuracy:.2f}%  ({correct}/{total} correct)")
        return accuracy


# -----------------------------------------------------------
# MAIN — Run training
# -----------------------------------------------------------
if __name__ == '__main__':

    print("\nCS731 Emotion Recognition Training")
    print("====================================")
    print(f"Selected model : {MODEL_NAME}")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Learning rate  : {LEARNING_RATE}")
    print(f"Max epochs     : {NUM_EPOCHS}")
    print(f"Early stopping : {PATIENCE} epochs patience")
    print(f"Save path      : {SAVE_PATH}\n")

    trainer = Trainer(MODEL_NAME, SAVE_PATH)
    trainer.train(NUM_EPOCHS)
