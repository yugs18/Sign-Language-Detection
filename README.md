# Sign Language Recognition

## Overview
This project implements a real-time Indian Sign Language (ISL) recognition system using computer vision and deep learning. The system can detect and interpret hand gestures representing numbers (1-9) and letters (A-Z) of the alphabet.

## Key Features
- Real-time sign language detection using webcam
- Recognition of ASL alphabet letters (A-Z) and numbers (1-9)
- Interactive GUI application with text output
- Spelling suggestions and search functionality
- Hybrid LSTM-CNN model with attention mechanism

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam for real-time detection

### Setup
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Application
To start the sign language recognition application:
```
python predict/app.py
```

### Controls
- Click "Start Camera" or press 'c' to activate your webcam
- Perform sign language gestures in front of the camera
- Press 'q' to quit the application
- Use the "Clear Text" button to reset the recognized text

## Project Structure
```
├── collect_images/            # Data collection scripts
├── data_preprocessing/        # Data preparation scripts
├── training/                  # Model training scripts
├── predict/                   # Prediction and application
│   ├── app.py                 # GUI application
│   └── prediction.ipynb       # Testing notebook
├── model/                     # Saved model files
├── requirements.txt           # Project dependencies
```

## Model Architecture
The system uses a hybrid LSTM-CNN architecture with an attention mechanism to recognize sign language gestures. The model processes body and hand keypoints extracted using MediaPipe.

## Data Pipeline
1. **Data Collection**: Capture sign language gestures using webcam
2. **Preprocessing**: Extract body and hand keypoints, normalize data
3. **Training**: Train the hybrid model on the processed data
4. **Prediction**: Real-time recognition with post-processing
