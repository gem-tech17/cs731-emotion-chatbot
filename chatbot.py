import requests
import json
import os
from dotenv import load_dotenv

# ============================================================
# CS731 Emotion-Aware Chatbot
# Uses OpenRouter API with Claude Haiku 4.5
# Adapts responses based on detected facial emotion
# ============================================================

load_dotenv()  # Load OPENROUTER_API_KEY from .env file

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
API_KEY   = os.getenv("OPENROUTER_API_KEY")
MODEL     = "anthropic/claude-haiku-4-5"           # Claude Haiku 4.5 via OpenRouter
API_URL   = "https://openrouter.ai/api/v1/chat/completions"
MAX_HISTORY = 20  # Keep last 20 messages (10 user + 10 assistant)

# -----------------------------------------------------------
# EMOTION-AWARE SYSTEM PROMPTS
# Each emotion gets a different personality/tone for the AI
# This is the core feature of our emotion-aware chatbot
# -----------------------------------------------------------
EMOTION_PROMPTS = {
    'happy': (
        "You are a warm, enthusiastic AI assistant. The user appears happy and engaged. "
        "Match their positive energy — be upbeat, encouraging, and celebratory. "
        "Use light humor where appropriate and keep responses energetic and fun."
    ),
    'sad': (
        "You are a compassionate, gentle AI assistant. The user appears sad or upset. "
        "Be empathetic, supportive, and kind. Speak softly and reassuringly. "
        "Acknowledge their feelings, offer comfort, and be patient. "
        "Avoid overly cheerful language — be warm but calm."
    ),
    'angry': (
        "You are a calm, patient AI assistant. The user appears frustrated or angry. "
        "Stay composed and understanding. Do not be defensive or dismissive. "
        "Acknowledge their frustration, be concise and clear, and help de-escalate "
        "by being extra helpful and solution-focused."
    ),
    'fear': (
        "You are a reassuring, steady AI assistant. The user appears anxious or fearful. "
        "Be calm, clear, and grounding. Use simple language and avoid overwhelming them. "
        "Offer reassurance, break things into small steps, and be extra patient."
    ),
    'surprise': (
        "You are an engaging, curious AI assistant. The user appears surprised or intrigued. "
        "Match their curiosity — be enthusiastic and informative. "
        "Embrace the unexpected, add interesting context, and keep energy high."
    ),
    'disgust': (
        "You are a professional, respectful AI assistant. The user appears uncomfortable. "
        "Be straightforward, neutral, and respectful. Avoid anything that might "
        "add to their discomfort. Be helpful and get to the point quickly."
    ),
    'contempt': (
        "You are a confident, direct AI assistant. The user appears skeptical or dismissive. "
        "Be clear, factual, and efficient. Prove your value through quality responses. "
        "Avoid being overly enthusiastic — be professional and precise."
    ),
    'neutral': (
        "You are a helpful, smart, and concise AI assistant. "
        "Be balanced, informative, and professional. "
        "Adapt your tone to whatever the user needs."
    )
}

# Default prompt when no emotion is detected
DEFAULT_PROMPT = EMOTION_PROMPTS['neutral']


# -----------------------------------------------------------
# CHATBOT CLASS
# -----------------------------------------------------------
class EmotionAwareChatbot:
    def __init__(self):
        """
        Initialise the chatbot with neutral emotion and empty history.
        """
        self.current_emotion = 'neutral'     # Default emotion
        self.conversation_history = []        # Stores chat messages
        self._update_system_prompt()          # Set initial system prompt

        if not API_KEY:
            print("WARNING: OPENROUTER_API_KEY not found in .env file")
        else:
            print("Chatbot initialised with Claude Haiku 4.5 via OpenRouter")
            print(f"Current emotion: {self.current_emotion}")

    def _update_system_prompt(self):
        """
        Update the system prompt based on current detected emotion.
        Called whenever emotion changes.
        """
        prompt = EMOTION_PROMPTS.get(self.current_emotion, DEFAULT_PROMPT)
        self.system_prompt = {
            "role": "system",
            "content": prompt
        }

    def set_emotion(self, emotion):
        """
        Update the chatbot's current emotion. Called by GUI when
        a new emotion is detected from the webcam.

        Args:
            emotion (str): Detected emotion label e.g. 'happy', 'sad'
        """
        if emotion != self.current_emotion:
            self.current_emotion = emotion
            self._update_system_prompt()
            print(f"Chatbot emotion updated: {emotion}")

    def send_message(self, user_message):
        """
        Send a user message to Claude Haiku 4.5 and get a response.
        Includes full conversation history for context.

        Args:
            user_message (str): The user's text input

        Returns:
            str: The AI's response text, or an error message
        """
        if not API_KEY:
            return "Error: No API key found. Please add OPENROUTER_API_KEY to your .env file."

        # Trim history to avoid token limits — keep last MAX_HISTORY messages
        if len(self.conversation_history) > MAX_HISTORY:
            self.conversation_history = self.conversation_history[-MAX_HISTORY:]

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Build full message list: system prompt + conversation history
        messages = [self.system_prompt] + self.conversation_history

        try:
            # Send request to OpenRouter API
            response = requests.post(
                url=API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": MODEL,
                    "messages": messages,
                    "max_tokens": 300    # Keep responses concise for GUI display
                }),
                timeout=30  # 30 second timeout
            )

            # Parse the response
            response_data = response.json()

            # Extract AI response text
            ai_response = response_data['choices'][0]['message']['content']

            # Add AI response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_response
            })

            return ai_response

        except KeyError:
            # API returned unexpected format — print raw response for debugging
            error_msg = f"API Error: {response.json()}"
            print(error_msg)
            return "Sorry, I encountered an error. Please try again."

        except requests.exceptions.Timeout:
            return "Request timed out. Please check your internet connection and try again."

        except requests.exceptions.ConnectionError:
            return "Connection error. Please check your internet connection."

        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def clear_history(self):
        """
        Clear conversation history. Called when starting a new session.
        """
        self.conversation_history = []
        print("Conversation history cleared.")


# -----------------------------------------------------------
# COMMAND LINE TEST — run directly to test chatbot
# (GUI will import this class instead of running directly)
# -----------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("CS731 Emotion-Aware Chatbot — Terminal Test")
    print("=" * 50)
    print("Commands: 'exit' to quit, 'emotion:<name>' to change emotion")
    print("Emotions: happy, sad, angry, fear, surprise, disgust, contempt, neutral")
    print("=" * 50)

    bot = EmotionAwareChatbot()
    print(f"\nCurrent emotion: {bot.current_emotion}")
    print("Type your message below:\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        # Quit command
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        # Emotion change command for testing — e.g. "emotion:happy"
        if user_input.lower().startswith('emotion:'):
            new_emotion = user_input.split(':')[1].strip().lower()
            if new_emotion in EMOTION_PROMPTS:
                bot.set_emotion(new_emotion)
                print(f"Emotion changed to: {new_emotion}")
            else:
                print(f"Unknown emotion. Choose from: {list(EMOTION_PROMPTS.keys())}")
            continue

        print("AI thinking...")
        response = bot.send_message(user_input)
        print(f"\nAI ({bot.current_emotion}): {response}\n")
