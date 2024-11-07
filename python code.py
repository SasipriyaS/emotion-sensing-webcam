import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained emotion detection model
model = load_model('C:/Users/DELL/Desktop/webcam/model_v6_23.hdf5')

# Define the list of emotion labels
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Create a function to detect emotions
def detect_emotion():
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera (webcam)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame (resize, convert to grayscale, normalize)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray, (48, 48))
        normalized_frame = resized_frame / 255.0
        reshaped_frame = np.reshape(normalized_frame, (1, 48, 48, 1))

        # Predict the emotion
        emotion_scores = model.predict(reshaped_frame)
        predicted_emotion_index = np.argmax(emotion_scores)
        predicted_emotion = emotions[predicted_emotion_index]

        # Display the detected emotion
        emotion_label.config(text=f"Detected Emotion: {predicted_emotion}")

        # Display the video frame in the Tkinter window
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        video_label.config(image=frame_tk)
        video_label.image = frame_tk

        root.update()  # Update the Tkinter window to display changes

        #if cv2.waitKey(1) & 0xFF == ord('q'):
           # break

    cap.release()
    cv2.destroyAllWindows()

# Create the main GUI window
root = tk.Tk()
root.title("Emotion Detection")

# Create a label for displaying detected emotions
emotion_label = tk.Label(root, text="Detected Emotion: ")
emotion_label.pack()

# Create a label for displaying the video feed
video_label = tk.Label(root)
video_label.pack()

# Create a button to start emotion detection
start_button = tk.Button(root, text="Start Detection", command=detect_emotion)
start_button.pack()

# Start the Tkinter main loop
root.mainloop()
