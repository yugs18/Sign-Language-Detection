import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from collections import Counter
from autocorrect import Speller
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import queue
import os

# Register custom function for model loading
@tf.keras.utils.register_keras_serializable()
def sum_over_time(x):
    return tf.keras.backend.sum(x, axis=1)

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition with Search")
        self.root.geometry("900x700")  # Adjusted size
        self.root.configure(bg="#f0f0f0")
        
        # Initialize variables
        self.is_webcam_active = False
        self.webcam_thread = None
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)
        self.prediction_buffer = []
        self.window_size = 10
        self.current_sentence = ""
        self.last_letter_time = time.time()
        self.no_prediction_count = 0
        self.space_threshold = 2  # 2 seconds of no prediction means space
        self.speller = Speller()
        self.current_word = ""
        self.spelling_suggestions = []
        
        # Parameters matching training pipeline
        self.BODY_KEYPOINTS = 33 * 3  # (x, y, z) for each body keypoint
        self.HAND_KEYPOINTS = 42 * 3  # (x, y, z) for each hand keypoint
        self.MAX_LEN = self.BODY_KEYPOINTS + self.HAND_KEYPOINTS
        
        # Load model and label mapping
        try:
            # Check if model files exist before loading
            if os.path.exists("model/lstm_cnn_model.keras") and os.path.exists("label_encoder.npy"):
                self.model = load_model("model/lstm_cnn_model.keras", 
                                       custom_objects={"sum_over_time": sum_over_time})
                self.label_classes = np.load("label_encoder.npy", allow_pickle=True)
                self.idx2label = {i: label for i, label in enumerate(self.label_classes)}
                print("Model and labels loaded successfully")
            else:
                raise FileNotFoundError("Model files not found")
        except Exception as e:
            print(f"Error loading model or labels: {e}")
            self.model = None
            self.label_classes = []
            self.idx2label = {}
            # Show error message to user after UI is created
            self.model_error = str(e)
        
        # Initialize Mediapipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                        max_num_hands=2,
                                        min_detection_confidence=0.5)
        
        # Create UI elements
        self.create_ui()
        
        # Setup keyboard shortcuts
        self.setup_keyboard_shortcuts()
        
        # Show model error message if any
        if hasattr(self, 'model_error'):
            self.root.after(1000, lambda: messagebox.showerror("Model Loading Error", 
                                                            f"Failed to load model: {self.model_error}\n\n"
                                                            f"Please ensure model files exist in the correct location."))
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for the application"""
        self.root.bind('c', lambda event: self.toggle_webcam())
        self.root.bind('q', lambda event: self.quit_application())
    
    def quit_application(self):
        """Clean up resources and quit the application"""
        self.stop_event.set()
        if hasattr(self, 'webcam_thread') and self.webcam_thread:
            self.webcam_thread.join(timeout=1.0)
        # Release MediaPipe resources
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'hands'):
            self.hands.close()
        self.root.quit()
    
    def create_ui(self):
        # Create paned window to allow user resizing of sections
        self.paned_window = tk.PanedWindow(self.root, orient=tk.VERTICAL, bg="#f0f0f0")
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame for camera and controls
        top_frame = tk.Frame(self.paned_window, bg="#f0f0f0")
        self.paned_window.add(top_frame, height=400)
        
        # Bottom frame for text displays
        bottom_frame = tk.Frame(self.paned_window, bg="#f0f0f0")
        self.paned_window.add(bottom_frame)
        
        # Title and camera controls in the top frame
        header_frame = tk.Frame(top_frame, bg="#f0f0f0")
        header_frame.pack(fill=tk.X)
        
        # Title
        title_label = tk.Label(header_frame, text="Sign Language Recognition", 
                              font=("Arial", 18, "bold"), bg="#f0f0f0")
        title_label.pack(side=tk.LEFT, pady=5)
        
        # Camera controls on the right side of header
        camera_controls = tk.Frame(header_frame, bg="#f0f0f0")
        camera_controls.pack(side=tk.RIGHT)
        
        # Camera button - Now using text instead of emoji
        self.camera_button = tk.Button(camera_controls, text="Start Camera (c)", 
                                     font=("Arial", 10), command=self.toggle_webcam, 
                                     bg="#4CAF50", fg="white", width=12)
        self.camera_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(camera_controls, text="Camera: Inactive", 
                                    font=("Arial", 10), bg="#f0f0f0", fg="#777777")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Main content area in the top frame - split into left (video) and right (controls)
        content_frame = tk.Frame(top_frame, bg="#f0f0f0")
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Video frame with reduced size
        video_container = tk.Frame(content_frame, bg="#f0f0f0")
        video_container.pack(side=tk.LEFT, padx=5)
        
        self.video_frame = tk.Label(video_container, bg="black", width=640, height=350)
        self.video_frame.pack()
        
        # Control panel on right side
        control_panel = tk.Frame(content_frame, bg="#f0f0f0")
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # Current prediction display
        prediction_frame = tk.LabelFrame(control_panel, text="Current Prediction", 
                                       font=("Arial", 10, "bold"), bg="#f0f0f0")
        prediction_frame.pack(fill=tk.X, pady=5)
        
        self.prediction_label = tk.Label(prediction_frame, text="None", 
                                       font=("Arial", 12, "bold"), bg="#f0f0f0", 
                                       fg="green", height=2)
        self.prediction_label.pack(fill=tk.X, padx=5)
        
        # Control buttons
        buttons_frame = tk.LabelFrame(control_panel, text="Controls", 
                                    font=("Arial", 10, "bold"), bg="#f0f0f0")
        buttons_frame.pack(fill=tk.X, pady=5)
        
        # Clear button
        self.clear_button = tk.Button(buttons_frame, text="Clear Text", 
                                    font=("Arial", 10), command=self.clear_text, 
                                    bg="#FF5722", fg="white", width=12)
        self.clear_button.pack(fill=tk.X, padx=5, pady=2)
        
        # Quit button
        quit_button = tk.Button(buttons_frame, text="Quit App (q)", 
                              font=("Arial", 10), command=self.quit_application, 
                              bg="#F44336", fg="white", width=12)
        quit_button.pack(fill=tk.X, padx=5, pady=2)
        
        # Text display areas in the bottom frame using a notebook for tabs
        notebook = ttk.Notebook(bottom_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Input and suggestions
        input_tab = tk.Frame(notebook, bg="#f0f0f0")
        notebook.add(input_tab, text="Input")
        
        # Current Word Frame in input tab
        word_frame = tk.LabelFrame(input_tab, text="Current Word", 
                                  font=("Arial", 10, "bold"), bg="#f0f0f0")
        word_frame.pack(pady=5, fill=tk.X)
        
        self.word_display = tk.Text(word_frame, height=1, font=("Arial", 12),
                                   wrap=tk.WORD, bd=2, relief=tk.GROOVE)
        self.word_display.pack(fill=tk.X, pady=5, padx=5)
        
        # Spelling Suggestions Frame in input tab
        suggestions_frame = tk.LabelFrame(input_tab, text="Spelling Suggestions", 
                                         font=("Arial", 10, "bold"), bg="#f0f0f0")
        suggestions_frame.pack(pady=5, fill=tk.X)
        
        self.suggestions_display = tk.Text(suggestions_frame, height=1, font=("Arial", 12),
                                          wrap=tk.WORD, bd=2, relief=tk.GROOVE)
        self.suggestions_display.pack(fill=tk.X, pady=5, padx=5)
        
        # Complete Sentences Frame in input tab
        sentence_frame = tk.LabelFrame(input_tab, text="Complete Sentences", 
                                      font=("Arial", 10, "bold"), bg="#f0f0f0")
        sentence_frame.pack(pady=5, fill=tk.X, expand=True)
        
        self.sentence_display = tk.Text(sentence_frame, height=3, font=("Arial", 12),
                                       wrap=tk.WORD, bd=2, relief=tk.GROOVE)
        self.sentence_display.pack(fill=tk.X, pady=5, padx=5, expand=True)
        
        # Tab 2: Search
        search_tab = tk.Frame(notebook, bg="#f0f0f0")
        notebook.add(search_tab, text="Search")
        
        # Search area
        search_frame = tk.Frame(search_tab, bg="#f0f0f0", pady=10)
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(search_frame, text="Search:", font=("Arial", 12), 
                bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(search_frame, textvariable=self.search_var, 
                                   font=("Arial", 12), bd=2, relief=tk.GROOVE)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Search button
        search_button = tk.Button(search_frame, text="Search", font=("Arial", 10), 
                                command=self.perform_search, bg="#2196F3", fg="white", 
                                width=10)
        search_button.pack(side=tk.LEFT, padx=5)
        
        # Tab 3: Instructions
        help_tab = tk.Frame(notebook, bg="#f0f0f0")
        notebook.add(help_tab, text="Help")
        
        # Instructions
        instructions = """Instructions:
1. Press 'c' to toggle camera on/off
2. Make sign language gestures to form words
3. Wait 2 seconds without gestures to add a space
4. Press Enter or click Search to search Google
5. You can also type directly in the search bar
6. Press 'q' to quit the application"""
        
        tk.Label(help_tab, text=instructions, font=("Arial", 11), 
                bg="#f0f0f0", justify=tk.LEFT).pack(pady=10, padx=10, fill=tk.BOTH)
        
        # Bind Enter key to search
        self.root.bind('<Return>', lambda event: self.perform_search())
    
    def clear_text(self):
        """Clear the current text and reset tracking variables"""
        self.current_sentence = ""
        self.current_word = ""
        self.spelling_suggestions = []
        self.update_displays()
        self.prediction_buffer = []
    
    def toggle_webcam(self):
        if self.is_webcam_active:
            # Stop webcam
            self.stop_event.set()
            if self.webcam_thread:
                self.webcam_thread.join(timeout=1.0)  # Add timeout to avoid hanging
            self.is_webcam_active = False
            self.camera_button.config(bg="#4CAF50", text="Start Camera (c)")
            self.status_label.config(text="Camera: Inactive", fg="#777777")
            # Reset video frame to blank black
            self.video_frame.config(image='')
        else:
            # Check if model is loaded before starting webcam
            if self.model is None:
                messagebox.showerror("Error", "Model not loaded. Cannot start webcam.")
                return
                
            # Start webcam
            self.stop_event.clear()
            self.webcam_thread = threading.Thread(target=self.webcam_loop)
            self.webcam_thread.daemon = True
            self.webcam_thread.start()
            self.is_webcam_active = True
            self.camera_button.config(bg="#F44336", text="Stop Camera (c)")
            self.status_label.config(text="Camera: Active", fg="green")
    
    def webcam_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            self.root.after(0, lambda: messagebox.showerror("Webcam Error", 
                                                         "Could not open webcam. Please check your camera connection."))
            self.is_webcam_active = False
            self.root.after(0, lambda: self.camera_button.config(bg="#4CAF50", text="Start Camera (c)"))
            self.root.after(0, lambda: self.status_label.config(text="Camera: Error", fg="red"))
            return
        
        prev_time = time.time()
        fps_update_time = time.time()
        fps_values = []
        
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip the image for a mirror effect
                frame = cv2.flip(frame, 1)
                
                # Extract keypoints and hand flag from the frame
                keypoints_vector, hand_present = self.extract_keypoints_from_frame(frame)
                
                # Process prediction
                if not hand_present:
                    predicted_label = "No Hand Detected"
                    self.no_prediction_count += 1
                else:
                    # Reshape to match the model's expected input shape: (1, MAX_LEN, 1)
                    input_data = keypoints_vector.reshape(1, self.MAX_LEN, 1)
                    
                    if self.model is not None:
                        try:
                            pred_prob = self.model.predict(input_data, verbose=0)
                            pred_class = np.argmax(pred_prob, axis=1)[0]
                            current_prediction = self.idx2label.get(pred_class, "Unknown")
                            
                            # Append prediction to the buffer
                            self.prediction_buffer.append(current_prediction)
                            if len(self.prediction_buffer) > self.window_size:
                                self.prediction_buffer.pop(0)
                            
                            # Compute majority vote over the buffer for smoothing
                            if self.prediction_buffer:
                                predicted_label = Counter(self.prediction_buffer).most_common(1)[0][0]
                                self.no_prediction_count = 0
                            else:
                                predicted_label = "Unknown"
                                self.no_prediction_count += 1
                        except Exception as e:
                            print(f"Prediction error: {e}")
                            predicted_label = "Prediction Error"
                            self.no_prediction_count += 1
                    else:
                        predicted_label = "Model not loaded"
                        self.no_prediction_count += 1
                
                # Update prediction label (using after method to ensure thread safety)
                self.root.after(0, lambda label=predicted_label: self.update_prediction(label))
                
                # Check if we need to add a letter or space
                current_time = time.time()
                if current_time - self.last_letter_time >= 1 and predicted_label not in ["No Hand Detected", "Unknown", "Model not loaded", "Prediction Error"]:
                    # Add letter to current word
                    self.current_word += predicted_label
                    self.last_letter_time = current_time
                    
                    # Generate spelling suggestions whenever the word changes
                    self.generate_spelling_suggestions()
                    
                    self.root.after(0, self.update_displays)
                
                # Check for space (no prediction for space_threshold seconds)
                if self.no_prediction_count >= self.space_threshold * 30:  # Assuming ~30 fps
                    if self.current_word:
                        # Add the word to the sentence
                        if self.spelling_suggestions and self.spelling_suggestions[0] != self.current_word:
                            # Use the first suggestion if available
                            self.current_sentence += self.spelling_suggestions[0] + " "
                        else:
                            # Otherwise use the current word
                            self.current_sentence += self.current_word + " "
                            
                        self.current_word = ""
                        self.spelling_suggestions = []
                        self.no_prediction_count = 0
                        self.root.after(0, self.update_displays)
                
                # Calculate FPS
                current_time = time.time()
                frame_time = current_time - prev_time
                if frame_time > 0:  # Prevent division by zero
                    fps = 1 / frame_time
                    fps_values.append(fps)
                    
                    # Average FPS over the last second
                    if current_time - fps_update_time >= 1.0:
                        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
                        fps_values = []
                        fps_update_time = current_time
                else:
                    fps = 0
                    
                prev_time = current_time
                
                # Display prediction and FPS on the frame
                cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Draw hand detection indicator
                status_color = (0, 255, 0) if hand_present else (0, 0, 255)
                cv2.circle(frame, (frame.shape[1] - 30, 30), 10, status_color, -1)
                
                # Scale down frame for display if needed
                display_frame = cv2.resize(frame, (480, 360))
                
                # Convert to tkinter format and display
                try:
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame)
                    img = ImageTk.PhotoImage(image=img)
                    
                    # Update video frame (ensure we're still running before setting)
                    if not self.stop_event.is_set():
                        self.video_frame.config(image=img)
                        self.video_frame.image = img  # Keep a reference to prevent garbage collection
                except Exception as e:
                    print(f"Frame display error: {e}")
                    
                # Small sleep to reduce CPU usage
                time.sleep(0.01)
        
        except Exception as e:
            print(f"Error in webcam loop: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Webcam error: {e}"))
        finally:
            # Release resources
            cap.release()
            # Reset button state if needed
            if self.is_webcam_active:
                self.is_webcam_active = False
                self.root.after(0, lambda: self.camera_button.config(bg="#4CAF50", text="Start Camera (c)"))
                self.root.after(0, lambda: self.status_label.config(text="Camera: Inactive", fg="#777777"))
    
    def generate_spelling_suggestions(self):
        """Generate spelling suggestions for the current word"""
        if not self.current_word or len(self.current_word) < 2:
            self.spelling_suggestions = []
            return
        
        try:
            # Get the main suggestion from autocorrect
            main_suggestion = self.speller(self.current_word)
            
            # Simple method to generate alternative suggestions
            # This is a simplified version - a real implementation would use a spelling library
            # that offers multiple suggestions
            suggestions = [main_suggestion]
            
            # Add more suggestions if main suggestion is different from current word
            if main_suggestion != self.current_word:
                # Add a couple more variations (this is just an example)
                if len(main_suggestion) > 3:
                    # Substitute a character
                    var1 = main_suggestion[:1] + main_suggestion[2:]
                    suggestions.append(var1)
                
                # Another variation
                if len(main_suggestion) > 2:
                    # Swap two characters
                    var2 = main_suggestion[0] + main_suggestion[2] + main_suggestion[1] + main_suggestion[3:]
                    suggestions.append(var2)
            
            # Remove duplicates and limit to 3 suggestions
            self.spelling_suggestions = list(dict.fromkeys(suggestions))[:3]
        except Exception as e:
            print(f"Error generating spelling suggestions: {e}")
            self.spelling_suggestions = []
    
    def update_prediction(self, label):
        self.prediction_label.config(text=label)
    
    def update_displays(self):
        """Update all three display areas"""
        # 1. Update current word display
        self.word_display.delete(1.0, tk.END)
        self.word_display.insert(tk.END, self.current_word)
        
        # 2. Update spelling suggestions display
        self.suggestions_display.delete(1.0, tk.END)
        if self.spelling_suggestions:
            self.suggestions_display.insert(tk.END, " | ".join(self.spelling_suggestions))
        
        # 3. Update sentence display
        self.sentence_display.delete(1.0, tk.END)
        self.sentence_display.insert(tk.END, self.current_sentence)
        
        # Also update search var
        combined_text = self.current_sentence + self.current_word
        self.search_var.set(combined_text)
    
    def perform_search(self):
        search_query = self.search_var.get().strip()
        if search_query:
            try:
                # Format the query for Google search
                formatted_query = search_query.replace(' ', '+')
                search_url = f"https://www.google.com/search?q={formatted_query}"
                
                # Open in default browser
                webbrowser.open(search_url)
                
                # Clear the current word and update sentence
                if self.current_word:
                    self.current_sentence += self.current_word + " "
                    self.current_word = ""
                    self.spelling_suggestions = []
                    self.update_displays()
            except Exception as e:
                messagebox.showerror("Search Error", f"Error performing search: {e}")
    
    def normalize_keypoints(self, body, hands_kps):
        """
        Convert the detected keypoints into a fixed-length array.
        
        Args:
            body (list): List of body keypoints dictionaries, each with keys "x", "y", "z".
            hands_kps (list): List of hand keypoints dictionaries, each with keys "x", "y", "z".
        
        Returns:
            np.array: 1D numpy array of length MAX_LEN.
        """
        keypoints = []
        
        # Process body keypoints (expected 33)
        if body and len(body) > 0:
            for point in body:
                keypoints.extend([point.get("x", 0.0), point.get("y", 0.0), point.get("z", 0.0)])
        else:
            keypoints.extend([0] * self.BODY_KEYPOINTS)
            
        # Process hand keypoints (expected 42)
        if hands_kps and len(hands_kps) > 0:
            for point in hands_kps:
                keypoints.extend([point.get("x", 0.0), point.get("y", 0.0), point.get("z", 0.0)])
        else:
            keypoints.extend([0] * self.HAND_KEYPOINTS)
        
        # Ensure the keypoints vector is of fixed length
        if len(keypoints) < self.MAX_LEN:
            keypoints += [0] * (self.MAX_LEN - len(keypoints))
        elif len(keypoints) > self.MAX_LEN:
            keypoints = keypoints[:self.MAX_LEN]
        
        return np.array(keypoints, dtype=np.float32)
    
    def extract_keypoints_from_frame(self, frame):
        try:
            # Convert the frame to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with pose
            pose_results = self.pose.process(image_rgb)
            body_keypoints = []
            if pose_results.pose_landmarks:
                for lm in pose_results.pose_landmarks.landmark:
                    body_keypoints.append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z
                    })
            
            # Process with hands
            hands_results = self.hands.process(image_rgb)
            hands_keypoints = []
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        hands_keypoints.append({
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z
                        })
            
            # If no hand keypoints are detected, mark hand_present as False.
            hand_present = len(hands_keypoints) > 0
            
            normalized = self.normalize_keypoints(body_keypoints, hands_keypoints)
            return normalized, hand_present
            
        except Exception as e:
            print(f"Error extracting keypoints: {e}")
            # Return empty arrays in case of error
            return np.zeros(self.MAX_LEN, dtype=np.float32), False
    
    def __del__(self):
        """Clean up resources when the application is closed"""
        self.stop_event.set()
        if hasattr(self, 'webcam_thread') and self.webcam_thread:
            self.webcam_thread.join(timeout=1.0)
        
        # Release MediaPipe resources
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'hands'):
            self.hands.close()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SignLanguageApp(root)
        root.protocol("WM_DELETE_WINDOW", app.quit_application)  # Ensure clean exit
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        # Show error in a messagebox if possible
        try:
            messagebox.showerror("Critical Error", f"Application failed to start: {e}")
        except:
            pass