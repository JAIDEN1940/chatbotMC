import pytesseract 
import pyautogui
import time
import cv2
import numpy as np
from PIL import Image, ImageGrab
import ollama  # Import Ollama for LLaMA model
import keyboard  # Import the keyboard module for key press detection
import os
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

# Set the path for Tesseract if it's not set globally
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Correct model path setup
models_path = r"C:\Users\\.ollama\models"  # Path to models
model_name = ""  # Use the correct model name
full_model_path = os.path.join(models_path, model_name)

# Log model path for validation
logging.info(f"Using model located at: {full_model_path}")
print(f"Using model located at: {full_model_path}")

# Ollama API setup (use the model directory path)
if not os.path.exists(full_model_path):
    logging.error(f"Model path {full_model_path} does not exist! Please download the model first.")
    print(f"Model path {full_model_path} does not exist! Please download the model first.")
    exit(1)

# Initialize sentiment analyzer for emotion detection
analyzer = SentimentIntensityAnalyzer()

# Set up logging to track bot interactions
logging.basicConfig(filename="chatbot_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

# Emotion detection function
def detect_emotion(player_message):
    sentiment_score = analyzer.polarity_scores(player_message)
    if sentiment_score['compound'] >= 0.05:
        return 'positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Adjust response based on sentiment/emotion of the player
def get_emotion_based_response(player_message, player_name):
    emotion = detect_emotion(player_message)
    
    if emotion == 'positive':
        return f"Wow, {player_name}! You seem really excited! {player_message} sounds great!"
    elif emotion == 'negative':
        return f"Oh no, {player_name}, it sounds like you're having a tough time. Don't worry, everything will get better!"
    else:
        return f"{player_name}, seems like you're feeling neutral. But hey, I’m here to chat anytime!"

# Get LLaMA response to the player's message
def get_llama_response(player_message, player_name):
    # Get emotion-based response
    emotion_response = get_emotion_based_response(player_message, player_name)

    # Prepare the prompt with emotion-based response
    prompt = f"Player {player_name} said: '{player_message}'. The player's mood is {emotion_response}"

    # Prepare the message structure for Ollama API
    messages = [
        {"role": "system", "content": "Your task is to respond to Minecraft players in a friendly and helpful way."},
        {"role": "user", "content": prompt},
    ]

    logging.info(f"Sending message to LLaMA model: {prompt}")

    try:
        # Use Ollama LLaMA model for generating the response
        response = ollama.chat(model=full_model_path, messages=messages)
        logging.info(f"Response from LLaMA model: {response}")
        return response["text"]
    except Exception as e:
        logging.error(f"Error generating response from LLaMA model: {e}")
        return "Sorry, I couldn't process that request."

# Capture screenshot of the area (bbox) or full screen
def capture_screen(bbox=None):
    """Capture a screenshot of the specified area (bbox) or the full screen."""
    try:
        img = ImageGrab.grab(bbox)  # Capture the screenshot
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
        return img_np
    except Exception as e:
        logging.error(f"Error capturing screen: {e}")
        return None

# Preprocess image for OCR
def preprocess_image(image):
    """Preprocess the image for faster OCR."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        return None

# Extract text from image using Tesseract OCR
def extract_text_from_image(image):
    """Use OCR to extract text from the image."""
    try:
        processed_img = preprocess_image(image)  # Preprocess image for faster OCR
        if processed_img is None:
            logging.error("Image preprocessing failed.")
            return ""
        text = pytesseract.image_to_string(processed_img)
        logging.info(f"Extracted text: {text}")
        return text
    except Exception as e:
        logging.error(f"Error extracting text with Tesseract: {e}")
        return ""

# Send a chat message in Minecraft using PyAutoGUI
def send_chat_message(message):
    """Send a chat message in the game using pyautogui."""
    try:
        pyautogui.press('t')  # Open the chat by pressing 't'
        time.sleep(0.3)  # Wait for the chat window to open
        pyautogui.write(f"<chat_bot> {message}")  # Type the AI-generated message
        pyautogui.press('enter')  # Press Enter to send the message
    except Exception as e:
        logging.error(f"Error sending chat message: {e}")

# Find player name from the OCR text (e.g., "playername>" instead of "<playername>")
def find_player_name_in_text(text):
    """Find the player name from the OCR text."""
    if '>' in text:
        # Extract the player name by finding the last word before '>'
        player_name = text.split('>')[0].strip()
        logging.info(f"Detected player name: {player_name}")
        return player_name
    return None

# Main capture and process function
def capture_and_process_screen():
    """Continuously capture screen and process the text using OCR."""
    print("Chat capture is now active.")
    while True:
        try:
            # Capture the chat area (you can specify bbox to focus on the chat window)
            screenshot = capture_screen(bbox=(0, 0, 960, 540))  # Capture the chat box region
            if screenshot is None:
                continue  # Skip if there's an issue with the capture

            # Extract text from the captured screenshot
            text = extract_text_from_image(screenshot)
            if not text:
                continue  # Skip if no text extracted

            # Print the captured text to the console (for debugging purposes)
            logging.info(f"Captured Text: {text}")

            # Look for player names in the chat
            player_name = find_player_name_in_text(text)
            if player_name:
                logging.info(f"Detected player: {player_name}")

                # Get LLaMA response to the player's message considering emotion
                ai_response = get_llama_response(text, player_name)

                # Add a slight delay to simulate human-like response time
                time.sleep(random.uniform(1, 3))  # Random delay for realism

                # Send the AI's response in the chat
                send_chat_message(ai_response)
                logging.info(f"Sent LLaMA response: {ai_response}")

            time.sleep(1)  # Adjust this delay as needed for real-time processing
        except Exception as e:
            logging.error(f"Error in capture and processing: {e}")
            time.sleep(1)  # Short sleep to prevent tight error loop

def start_bot():
    """Start the chat capture loop."""
    capture_and_process_screen()

def stop_bot():
    """Stop the chat capture loop (to be implemented, for now, it just stops the script)."""
    print("Stopping the bot.")
    exit()  # Exit the script (you can modify this to implement a safer shutdown if needed)

def toggle_script():
    """Monitor keyboard presses to start/stop the bot."""
    print("Press 'Q' to start the bot and 'E' to stop the bot.")
    while True:
        if keyboard.is_pressed('q'):  # If 'Q' is pressed, start the bot
            start_bot()  # Start capture loop
            break  # Exit the loop once the script is started
        elif keyboard.is_pressed('e'):  # If 'E' is pressed, stop the bot
            stop_bot()  # Stop the script
            break  # Exit the loop once the script is stopped
        time.sleep(0.1)  # Small delay to prevent excessive CPU usage

if __name__ == "__main__":
    # Start the monitoring of keypresses for the main script
    toggle_script()
