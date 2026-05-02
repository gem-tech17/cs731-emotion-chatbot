import sys
import os
import cv2
import speech_recognition as sr
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QImage, QPixmap

from inference import Inferencer
from chatbot import EmotionAwareChatbot

# ============================================================
# CS731 Emotion-Aware Chatbot — PyQt5 Desktop GUI
# Layout: Left = Webcam + Emotion | Right = Chat Interface
# Features: YOLOv8 face detection, emotion smoothing,
#           voice input, text input, Claude Haiku 4.5
# ============================================================

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
MODEL_PATH       = "checkpoints/efficientnet/best.pt"  # Best trained model
YOLO_PATH        = "yolov8n-face.pt"                   # YOLOv8 face detector
WEBCAM_INDEX     = 0                                    # Default webcam
EMOTION_INTERVAL = 5                                   # Run detection every N frames
SMOOTH_WINDOW    = 8                                    # Smooth over last N detections

# Emotion colours for the UI label
EMOTION_COLORS = {
    'happy':    '#FFD700',
    'sad':      '#6495ED',
    'angry':    '#FF4444',
    'fear':     '#9B59B6',
    'surprise': '#FF8C00',
    'disgust':  '#2ECC71',
    'contempt': '#95A5A6',
    'neutral':  '#ECF0F1',
}


# -----------------------------------------------------------
# WORKER: Chatbot API call in background thread
# -----------------------------------------------------------
class ChatWorker(QObject):
    response_ready = pyqtSignal(str)

    def __init__(self, chatbot, message):
        super().__init__()
        self.chatbot = chatbot
        self.message = message

    def run(self):
        """Send message to Claude Haiku 4.5 and emit response."""
        response = self.chatbot.send_message(self.message)
        self.response_ready.emit(response)


# -----------------------------------------------------------
# WORKER: Voice recognition in background thread
# -----------------------------------------------------------
class VoiceWorker(QObject):
    text_ready = pyqtSignal(str)
    error      = pyqtSignal(str)

    def run(self):
        """Listen to microphone and convert speech to text."""
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            self.text_ready.emit(text)
        except sr.WaitTimeoutError:
            self.error.emit("No speech detected. Please try again.")
        except sr.UnknownValueError:
            self.error.emit("Could not understand audio. Please try again.")
        except sr.RequestError:
            self.error.emit("Speech recognition service unavailable.")
        except Exception as e:
            self.error.emit(f"Microphone error: {str(e)}")


# -----------------------------------------------------------
# MAIN GUI WINDOW
# -----------------------------------------------------------
class EmotionChatbotGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CS731 — Emotion-Aware Chatbot")
        self.setMinimumSize(1100, 700)
        self.resize(1200, 750)
        self.setStyleSheet(self._get_stylesheet())

        # ---------------------------------------------------
        # Load YOLOv8 face detector
        # ---------------------------------------------------
        self.face_model = None
        if os.path.exists(YOLO_PATH):
            try:
                self.face_model = YOLO(YOLO_PATH)
                print("YOLOv8 face model loaded ✓")
            except Exception as e:
                print(f"Could not load YOLO model: {e}")
        else:
            print(f"YOLO model not found at {YOLO_PATH}")

        # ---------------------------------------------------
        # Load emotion classifier
        # ---------------------------------------------------
        self.inferencer = None
        if os.path.exists(MODEL_PATH):
            try:
                self.inferencer = Inferencer(MODEL_PATH)
                print(f"Emotion model loaded: {MODEL_PATH}")
            except Exception as e:
                print(f"Could not load emotion model: {e}")
        else:
            print(f"Model not found at {MODEL_PATH}")

        # ---------------------------------------------------
        # Initialise chatbot
        # ---------------------------------------------------
        self.chatbot = EmotionAwareChatbot()

        # ---------------------------------------------------
        # State variables
        # ---------------------------------------------------
        self.current_emotion = 'neutral'
        self.frame_count     = 0
        self.is_listening    = False
        self.chat_thread     = None
        self.voice_thread    = None
        self.emotion_history = []  # For smoothing
        self.last_auto_message_emotion = None  # Track last auto-messaged emotion

        # ---------------------------------------------------
        # Build UI
        # ---------------------------------------------------
        self._build_ui()

        # ---------------------------------------------------
        # Start webcam with MJPG format (required for WSL)
        # ---------------------------------------------------
        self.cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self._add_chat_message("System", "Webcam not found. Please check connection.")
        else:
            print("Webcam opened successfully ✓")

        # Timer: update frame every 30ms
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(30)

        self._add_chat_message(
            "System",
            "Welcome! I can detect your emotions and respond accordingly. "
            "Type a message or click the microphone to speak!"
        )

    # -----------------------------------------------------------
    # UI BUILDER
    # -----------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # LEFT PANEL
        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_panel.setFixedWidth(480)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        cam_title = QLabel("📷  Live Emotion Detection")
        cam_title.setObjectName("panelTitle")
        cam_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(cam_title)

        self.webcam_label = QLabel()
        self.webcam_label.setObjectName("webcamLabel")
        self.webcam_label.setFixedSize(460, 345)
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setText("Initialising webcam...")
        left_layout.addWidget(self.webcam_label, alignment=Qt.AlignCenter)

        emotion_frame = QFrame()
        emotion_frame.setObjectName("emotionFrame")
        emotion_layout = QVBoxLayout(emotion_frame)
        emotion_layout.setSpacing(4)

        emotion_header = QLabel("DETECTED EMOTION")
        emotion_header.setObjectName("emotionHeader")
        emotion_header.setAlignment(Qt.AlignCenter)
        emotion_layout.addWidget(emotion_header)

        self.emotion_label = QLabel("😐  Neutral")
        self.emotion_label.setObjectName("emotionLabel")
        self.emotion_label.setAlignment(Qt.AlignCenter)
        emotion_layout.addWidget(self.emotion_label)

        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setObjectName("confidenceLabel")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        emotion_layout.addWidget(self.confidence_label)

        left_layout.addWidget(emotion_frame)
        left_layout.addStretch()

        # RIGHT PANEL
        right_panel = QFrame()
        right_panel.setObjectName("rightPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        chat_title = QLabel("💬  Chat with AI")
        chat_title.setObjectName("panelTitle")
        chat_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(chat_title)

        self.chat_display = QTextEdit()
        self.chat_display.setObjectName("chatDisplay")
        self.chat_display.setReadOnly(True)
        self.chat_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.chat_display)

        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        self.text_input = QLineEdit()
        self.text_input.setObjectName("textInput")
        self.text_input.setPlaceholderText("Type your message here...")
        self.text_input.returnPressed.connect(self._send_text_message)
        input_row.addWidget(self.text_input)

        self.voice_btn = QPushButton("🎤")
        self.voice_btn.setObjectName("voiceBtn")
        self.voice_btn.setFixedSize(48, 48)
        self.voice_btn.setToolTip("Click to speak")
        self.voice_btn.clicked.connect(self._start_voice_input)
        input_row.addWidget(self.voice_btn)

        self.send_btn = QPushButton("Send ➤")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.setFixedHeight(48)
        self.send_btn.clicked.connect(self._send_text_message)
        input_row.addWidget(self.send_btn)

        right_layout.addLayout(input_row)

        clear_btn = QPushButton("🗑  Clear Chat")
        clear_btn.setObjectName("clearBtn")
        clear_btn.clicked.connect(self._clear_chat)
        right_layout.addWidget(clear_btn)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

    # -----------------------------------------------------------
    # WEBCAM + EMOTION DETECTION
    # -----------------------------------------------------------
    def _update_frame(self):
        """Read webcam frame, run detection, update display."""
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_count += 1

        # Run YOLOv8 + emotion detection every N frames
        if self.inferencer and self.face_model and self.frame_count % EMOTION_INTERVAL == 0:
            self._detect_emotion(frame)

        # Convert BGR to RGB and display in Qt label
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch  = rgb_frame.shape
        qt_image  = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap    = QPixmap.fromImage(qt_image)
        scaled    = pixmap.scaled(
            self.webcam_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.webcam_label.setPixmap(scaled)

    def _detect_emotion(self, frame):
        """
        Detect faces with YOLOv8, classify emotion on each face crop.
        Uses a smoothing window to reduce flickering.
        """
        try:
            results = self.face_model(frame, verbose=False, conf=0.4)
            best_emotion = None
            best_conf    = 0.0

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])

                    # Padding around face
                    pad = 30
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(frame.shape[1], x2 + pad)
                    y2 = min(frame.shape[0], y2 + pad)

                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    # Predict emotion on face crop only
                    label, display, confidence, emoji = self.inferencer.predict(face_crop)

                    # Draw green bounding box + label on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{display}: {confidence:.0%}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2
                    )

                    if confidence > best_conf:
                        best_conf    = confidence
                        best_emotion = (label, display, confidence, emoji)

            if best_emotion:
                label, display, confidence, emoji = best_emotion

                # Smoothing: add to history, pick most common
                self.emotion_history.append(label)
                if len(self.emotion_history) > SMOOTH_WINDOW:
                    self.emotion_history.pop(0)
                smoothed = max(set(self.emotion_history), key=self.emotion_history.count)

                # Update UI emotion label
                color = EMOTION_COLORS.get(smoothed, '#ECF0F1')
                emoji_map = {
                    'happy': '😊', 'sad': '😢', 'angry': '😠',
                    'fear': '😨', 'surprise': '😲', 'disgust': '🤢',
                    'contempt': '😒', 'neutral': '😐'
                }
                self.emotion_label.setText(
                    f"{emoji_map.get(smoothed, '😐')}  {smoothed.capitalize()}"
                )
                self.emotion_label.setStyleSheet(
                    f"color: {color}; font-size: 28px; font-weight: bold;"
                )
                self.confidence_label.setText(f"Confidence: {confidence:.1%}")

                # Notify chatbot if emotion changed
                if smoothed != self.current_emotion:
                    self.current_emotion = smoothed
                    self.chatbot.set_emotion(smoothed)
                self._auto_emotion_message(smoothed)

        except Exception:
            pass

    # -----------------------------------------------------------
    # CHAT — Text input
    # -----------------------------------------------------------
    def _auto_emotion_message(self, emotion):
        """Automatically send emotion-based message when emotion changes."""
        messages = {
            "happy":    "I can see you are happy! Great energy for studying. What topic shall we work on?",
            "sad":      "You seem sad. Would you like to talk about it, or shall I help you with your studies?",
            "angry":    "You seem frustrated. Let us take a breath and work through this together.",
            "fear":     "You look anxious. Do not worry, we will break everything into small steps!",
            "surprise": "You look surprised! Did something click for you?",
            "disgust":  "You seem uncomfortable. Want to try a different approach to this topic?",
            "contempt": "Feeling bored? Let me make this more interesting for you!",
            "neutral":  "Let me know if you'd like more explanation or pratice questions."
        }
        message = messages.get(emotion, "")
        if message and emotion != self.last_auto_message_emotion:
            self.last_auto_message_emotion = emotion
            self._add_chat_message("AI", message)
            self.chatbot.conversation_history.append({"role": "assistant", "content": message})

    def _send_text_message(self):
        message = self.text_input.text().strip()
        if not message:
            return
        self.text_input.clear()
        self._add_chat_message("You", message)
        self._set_input_enabled(False)
        self.status_label.setText("🤔  AI is thinking...")

        self.chat_thread = QThread()
        self.chat_worker = ChatWorker(self.chatbot, message)
        self.chat_worker.moveToThread(self.chat_thread)
        self.chat_thread.started.connect(self.chat_worker.run)
        self.chat_worker.response_ready.connect(self._on_response_ready)
        self.chat_worker.response_ready.connect(self.chat_thread.quit)
        self.chat_thread.start()

    def _on_response_ready(self, response):
        emoji_map = {
            'happy': '😊', 'sad': '😢', 'angry': '😠',
            'fear': '😨', 'surprise': '😲', 'disgust': '🤢',
            'contempt': '😒', 'neutral': '😐'
        }
        emoji = emoji_map.get(self.current_emotion, '🤖')
        self._add_chat_message(f"AI {emoji}", response)
        self._set_input_enabled(True)
        self.text_input.setFocus()
        self.status_label.setText("")

    # -----------------------------------------------------------
    # CHAT — Voice input
    # -----------------------------------------------------------
    def _start_voice_input(self):
        if self.is_listening:
            return
        self.is_listening = True
        self.voice_btn.setText("🔴")
        self.voice_btn.setStyleSheet("background-color: #FF4444; color: white;")
        self.status_label.setText("🎤  Listening... Speak now!")
        self._set_input_enabled(False)

        self.voice_thread = QThread()
        self.voice_worker = VoiceWorker()
        self.voice_worker.moveToThread(self.voice_thread)
        self.voice_thread.started.connect(self.voice_worker.run)
        self.voice_worker.text_ready.connect(self._on_voice_text_ready)
        self.voice_worker.error.connect(self._on_voice_error)
        self.voice_worker.text_ready.connect(self.voice_thread.quit)
        self.voice_worker.error.connect(self.voice_thread.quit)
        self.voice_thread.start()

    def _on_voice_text_ready(self, text):
        self.is_listening = False
        self.voice_btn.setText("🎤")
        self.voice_btn.setStyleSheet("")
        self._set_input_enabled(False)
        self._add_chat_message("🎤 You (voice)", text)
        self.status_label.setText("🤔  AI is thinking...")

        self.chat_thread = QThread()
        self.chat_worker = ChatWorker(self.chatbot, text)
        self.chat_worker.moveToThread(self.chat_thread)
        self.chat_thread.started.connect(self.chat_worker.run)
        self.chat_worker.response_ready.connect(self._on_response_ready)
        self.chat_worker.response_ready.connect(self.chat_thread.quit)
        self.chat_thread.start()

    def _on_voice_error(self, error_msg):
        self.is_listening = False
        self.voice_btn.setText("🎤")
        self.voice_btn.setStyleSheet("")
        self.status_label.setText(f"⚠️  {error_msg}")
        self._set_input_enabled(True)

    # -----------------------------------------------------------
    # CHAT HELPERS
    # -----------------------------------------------------------
    def _add_chat_message(self, sender, message):
        if sender.startswith("You"):
            color = "#64B5F6"
        elif sender.startswith("AI"):
            color = "#81C784"
        else:
            color = "#FFD54F"

        html = (
            f'<p style="margin: 6px 0;">'
            f'<span style="color: {color}; font-weight: bold;">{sender}:</span> '
            f'<span style="color: #ECEFF1;">{message}</span>'
            f'</p>'
        )
        self.chat_display.append(html)
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _clear_chat(self):
        self.chat_display.clear()
        self.chatbot.clear_history()
        self._add_chat_message("System", "Chat cleared. Starting fresh!")

    def _set_input_enabled(self, enabled):
        self.text_input.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        self.voice_btn.setEnabled(enabled)

    # -----------------------------------------------------------
    # STYLESHEET
    # -----------------------------------------------------------
    def _get_stylesheet(self):
        return """
            QMainWindow, QWidget {
                background-color: #1A1A2E;
                color: #ECEFF1;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QFrame#leftPanel, QFrame#rightPanel {
                background-color: #16213E;
                border-radius: 12px;
                padding: 8px;
            }
            QLabel#panelTitle {
                font-size: 15px;
                font-weight: bold;
                color: #90CAF9;
                padding: 6px;
                border-bottom: 1px solid #0F3460;
                margin-bottom: 4px;
            }
            QLabel#webcamLabel {
                background-color: #0D0D1A;
                border-radius: 8px;
                border: 2px solid #0F3460;
                color: #546E7A;
                font-size: 14px;
            }
            QFrame#emotionFrame {
                background-color: #0F3460;
                border-radius: 10px;
                padding: 10px;
                margin-top: 4px;
            }
            QLabel#emotionHeader {
                color: #90CAF9;
                font-size: 12px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QLabel#emotionLabel {
                font-size: 28px;
                font-weight: bold;
                color: #ECF0F1;
                padding: 8px;
            }
            QLabel#confidenceLabel {
                color: #B0BEC5;
                font-size: 12px;
            }
            QTextEdit#chatDisplay {
                background-color: #0D0D1A;
                border: 1px solid #0F3460;
                border-radius: 8px;
                padding: 10px;
                color: #ECEFF1;
                font-size: 13px;
            }
            QLabel#statusLabel {
                color: #FFD54F;
                font-size: 12px;
                font-style: italic;
                min-height: 20px;
            }
            QLineEdit#textInput {
                background-color: #0F3460;
                border: 1px solid #1565C0;
                border-radius: 8px;
                padding: 10px 14px;
                color: #ECEFF1;
                font-size: 13px;
                min-height: 28px;
            }
            QLineEdit#textInput:focus { border: 1px solid #42A5F5; }
            QPushButton#sendBtn {
                background-color: #1565C0;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 13px;
                min-width: 90px;
            }
            QPushButton#sendBtn:hover    { background-color: #1976D2; }
            QPushButton#sendBtn:pressed  { background-color: #0D47A1; }
            QPushButton#sendBtn:disabled { background-color: #37474F; color: #607D8B; }
            QPushButton#voiceBtn {
                background-color: #0F3460;
                color: white;
                border: 1px solid #1565C0;
                border-radius: 8px;
                font-size: 20px;
            }
            QPushButton#voiceBtn:hover    { background-color: #1565C0; }
            QPushButton#voiceBtn:disabled { background-color: #37474F; color: #607D8B; }
            QPushButton#clearBtn {
                background-color: transparent;
                color: #546E7A;
                border: 1px solid #37474F;
                border-radius: 6px;
                padding: 6px;
                font-size: 12px;
            }
            QPushButton#clearBtn:hover { background-color: #263238; color: #90A4AE; }
            QScrollBar:vertical {
                background: #0D0D1A;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #1565C0;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical { height: 0px; }
        """

    # -----------------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------------
    def closeEvent(self, event):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()


# -----------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("CS731 Emotion Chatbot")
    window = EmotionChatbotGUI()
    window.show()
    sys.exit(app.exec_())
