# Sign Language Detection Project Documentation

## Project Overview

This project implements a complete sign language recognition system that can detect and interpret American Sign Language (ASL) gestures in real-time. The system uses computer vision techniques and deep learning to recognize hand gestures representing numbers (1-9) and letters (A-Z) of the alphabet.

### Project Structure

```
Sign-Language-Recognition-main/
├── collect_images/            # Data collection scripts
│   ├── collect-data.ipynb     # Main script for collecting images via webcam
│   ├── image-counter.py       # Utility to count collected images
│   └── random_pics.ipynb      # Script to view random samples from dataset
├── data_preprocessing/        # Data preparation scripts
│   ├── augmentation.ipynb     # Image augmentation to expand dataset
│   ├── generate_keypoints.ipynb # Extract body and hand keypoints using MediaPipe
│   ├── preprocess.ipynb       # Convert keypoints to model-ready format
│   └── split_data.ipynb       # Split data into train/val/test sets
├── training/                  # Model training scripts
│   └── training.ipynb         # LSTM-CNN hybrid model with attention mechanism
├── predict/                   # Prediction and application scripts
│   ├── app.py                 # GUI application for real-time sign language detection
│   └── prediction.ipynb       # Notebook for testing prediction functionality
├── model/                     # Saved model files
│   ├── best_model.keras       # Best model from training
│   └── lstm_cnn_model.keras   # Final trained model
├── dataset/                   # Dataset directories
│   ├── images/                # Original collected images
│   ├── augmented_images/      # Augmented dataset
│   ├── keypoints/             # Extracted keypoints in JSON format
│   └── data/                  # Processed data splits (train/val/test)
├── dataset.npz                # Preprocessed dataset in NumPy format
├── label_encoder.npy          # Label encoder for class mapping
├── requirements.txt           # Project dependencies
└── sign-language-detection-complete-documentation.md # This documentation
```

### Workflow

1. **Data Collection**: Capture sign language gestures using webcam
2. **Data Preprocessing**:
   - Augment images to increase dataset diversity
   - Extract body and hand keypoints using MediaPipe
   - Split data into training, validation, and test sets
   - Normalize and prepare data for model training
3. **Model Training**: Train a hybrid LSTM-CNN model with attention mechanism
4. **Prediction**: Real-time sign language recognition with GUI interface

### Dependencies

The project requires the following main dependencies:
- tensorflow/keras (≥2.0.0): Deep learning framework
- mediapipe (≥0.8.0): For pose and hand landmark detection
- opencv-python (≥4.5.0): For image and video processing
- numpy (≥1.19.0): For numerical operations
- scikit-learn (≥0.24.0): For data preprocessing and evaluation
- autocorrect (≥2.0.0): For spelling correction in predictions
- Pillow (≥8.0.0): For image processing
- matplotlib/seaborn: For visualization
- tkinter: For GUI (included in Python standard library)

A complete list of dependencies is available in the `requirements.txt` file.

## Step 1: Data Collection

### Overview
This module is responsible for collecting image data of sign language gestures using a webcam. The system captures images of hand gestures representing numbers (1-9) and letters (A-Z) of the alphabet in sign language. These images will later be used to train a machine learning model to recognize and interpret sign language gestures.

### Dependencies
- OpenCV (`cv2`): For image capture and processing
- os: For file system operations
- time: For implementing delays between captures
- uuid: For generating unique identifiers for image filenames
- tensorflow/keras: For building and training neural network models
- numpy: For numerical operations and array handling
- mediapipe: For body pose and hand landmark detection
- PIL (Pillow): For image processing operations
- matplotlib/seaborn: For data visualization
- scikit-learn: For data preprocessing and evaluation metrics
- tqdm: For progress tracking during processing
- autocorrect: For spelling correction in the prediction application
- tkinter: For building the graphical user interface (GUI)

### Configuration
```python
IMAGES_PATH = "../dataset/images"  # Path where collected images will be saved
LABELS = list("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Sign language gestures to capture
NUMBER_OF_IMGS = 1000  # Number of images to collect per gesture
```

### Functions

#### `create_directory(path)`
Creates a directory at the specified path if it doesn't already exist.

**Parameters:**
- `path` (str): The path to the directory that needs to be created

**Example:**
```python
create_directory('images/collected')  # Creates the 'collected' directory inside 'images'
```

#### `capture_images_for_class(label, num_images, video_capture)`
Captures a specified number of images for a given sign language gesture label.

**Parameters:**
- `label` (str): The sign language gesture label (e.g., '1', 'A')
- `num_images` (int): Number of images to capture for this label
- `video_capture` (cv2.VideoCapture): The webcam capture object

**Features:**
- Creates a subdirectory for each label within the main images directory
- Provides a 5-second preparation time before capturing images
- Allows pausing/resuming the capture process using the spacebar
- Displays the webcam feed during capture
- Generates unique filenames for each captured image
- Allows early exit by pressing 'q'

#### `main()`
Orchestrates the entire data collection process.

**Process:**
1. Creates the main data directory
2. Initializes the webcam
3. Iterates through each sign language gesture label
4. Captures the specified number of images for each label
5. Releases resources upon completion

### Usage
Run the script directly to start the data collection process:
```
python data_collection.py
```

### User Interaction
- Press SPACEBAR to pause/resume capture
- Press 'q' to exit the capture process early

### Output
The script creates a directory structure as follows:
```
../dataset/images/
├── 1/
│   ├── 1_[uuid1].jpg
│   ├── 1_[uuid2].jpg
│   └── ...
├── 2/
│   └── ...
...
└── Z/
    └── ...
```
Each subdirectory contains approximately 1,000 images of the corresponding sign language gesture.

!![Original Images](doc-images\Picture1.png)

## Step 2: Data Preprocessing

### 2.1: Data Augmentation

#### Overview
This module performs data augmentation on the collected sign language gesture images. Data augmentation is a technique used to artificially expand the training dataset by creating modified versions of existing images, which helps improve model robustness and prevent overfitting.

#### Dependencies
- os: For file system operations
- tensorflow (tf): For image processing operations
- tensorflow.keras.preprocessing.image.ImageDataGenerator: For generating augmented images
- uuid: For generating unique identifiers for augmented image filenames

#### Configuration
```python
input_dir = "../dataset/images"         # Directory containing original images
output_dir = "../dataset/augmented_images"  # Directory for storing augmented images
labels = list("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Sign language gestures
```

#### Augmentation Parameters
The augmentation generator is configured with the following transformations:
- **rotation_range=30**: Randomly rotate images up to 30 degrees
- **width_shift_range=0.1**: Randomly shift images horizontally by up to 10%
- **height_shift_range=0.1**: Randomly shift images vertically by up to 10%
- **shear_range=0.2**: Apply shear transformations with an intensity of 0.2
- **zoom_range=0.2**: Randomly zoom in or out by up to 20%
- **horizontal_flip=True**: Randomly flip images horizontally
- **fill_mode='nearest'**: Fill empty spaces after transformations using the nearest pixel values

#### Functions

##### `augment_images(input_dir, output_dir, label, augment_count=5)`
Augments images for a specific label and saves them to the output directory.

**Parameters:**
- `input_dir` (str): Path to the directory containing original images
- `output_dir` (str): Path to the directory where augmented images will be saved
- `label` (str): The sign language gesture label (e.g., '1', 'A')
- `augment_count` (int): Number of augmented versions to create for each original image

**Process:**
1. Loads each original image from the input directory
2. Resizes the image to 300x300 pixels
3. Generates `augment_count` number of augmented versions using the defined transformations
4. Saves each augmented image with a unique UUID-based filename

#### Execution Flow
1. Creates the main output directory for augmented images
2. Creates subdirectories for each sign language gesture label
3. Initializes the ImageDataGenerator with the specified augmentation parameters
4. Processes each label's directory, applying augmentation to all images
5. For each original image, creates 5 augmented versions

#### Output
The script creates a directory structure as follows:
```
../dataset/augmented_images/
├── 1/
│   ├── [uuid1].jpg
│   ├── [uuid2].jpg
│   └── ...
├── 2/
│   └── ...
...
└── Z/
    └── ...
```
Each subdirectory contains augmented versions of the original sign language gesture images, with each augmented image having a unique UUID-based filename.

#### Benefits of Augmentation
- Increases the diversity of the training dataset
- Improves model's ability to generalize to unseen data
- Reduces overfitting by introducing variations of the same gesture
- Enhances model robustness to different hand positions, lighting conditions, and angles

![Augmentated Images](doc-images\Picture2.png)

### 2.2: Generating Keypoints

#### Overview
This module extracts body and hand keypoints from the augmented sign language gesture images using MediaPipe. These keypoints represent the position of various body joints and hand landmarks, which will serve as features for the machine learning model to recognize sign language gestures.

#### Dependencies
- os: For file system operations
- cv2 (OpenCV): For image loading and processing
- json: For saving keypoint data in structured format
- tqdm: For progress tracking during processing
- mediapipe: For body pose and hand landmark detection

#### Configuration
```python
IMAGE_PATH = '../dataset/augmented_images'  # Path to augmented images
OUTPUT_PATH = '../dataset/keypoints'       # Path to save extracted keypoints
```

#### MediaPipe Setup
The module initializes two MediaPipe solutions:
1. **Pose Detection**: For extracting body keypoints
   ```python
   mp_pose = mp.solutions.pose
   pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
   ```

2. **Hand Landmark Detection**: For extracting hand keypoints
   ```python
   mp_hands = mp.solutions.hands
   hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
   ```

#### Functions

##### `create_dir(directory_path)`
Creates a directory if it doesn't already exist.

**Parameters:**
- `directory_path` (str): Path to the directory to be created

##### `validate_json(output_path, label)`
Validates the JSON file for a given label and prints the total count of processed images.

**Parameters:**
- `output_path` (str): Path to the folder containing JSON files
- `label` (str): Label identifier (e.g., '1', 'A')

##### `extract_body_keypoints(image_path)`
Extracts body pose keypoints from an image using MediaPipe Pose.

**Parameters:**
- `image_path` (str): Path to the input image

**Returns:**
- List of 33 keypoints, each containing x, y, z coordinates
- Missing keypoints are padded with zeros to ensure consistent data structure

##### `extract_hand_keypoints(image_path)`
Extracts hand landmarks from an image using MediaPipe Hands.

**Parameters:**
- `image_path` (str): Path to the input image

**Returns:**
- List of up to 42 keypoints (21 per hand for up to two hands)
- Missing keypoints are padded with zeros to ensure consistent data structure

##### `process_images_with_validation(base_path, output_path)`
Processes all images in the dataset to extract keypoints and save them in JSON format.

**Parameters:**
- `base_path` (str): Path to the base folder containing image directories
- `output_path` (str): Path to the folder where JSON files will be saved

**Process:**
1. Creates the output directory if it doesn't exist
2. Initializes data structures for each label
3. For each label (1-9, A-Z):
   - Processes all images in the corresponding directory
   - Extracts body and hand keypoints for each image
   - Accumulates keypoint data for the label
4. Saves the accumulated data for each label to a JSON file
5. Validates the saved data and reports the total number of processed images

#### Keypoint Data Structure
The extracted keypoints are stored in JSON files with the following structure:
```json
[
  {
    "image_name": "filename.jpg",
    "keypoints": {
      "body": [
        {"x": 0.45, "y": 0.32, "z": 0.01},
        ...
        // 33 body keypoints total
      ],
      "hands": [
        {"x": 0.56, "y": 0.25, "z": 0.02},
        ...
        // 42 hand keypoints total (21 per hand × 2 hands)
      ]
    }
  },
  ...
]
```

#### Output
The script creates a directory (`../dataset/keypoints`) containing JSON files for each label (1-9, A-Z). Each JSON file contains the keypoint data for all images of that particular sign language gesture.

#### Benefits of Keypoint Extraction
- Reduces the dimensionality of the data compared to raw images
- Focuses on the essential features (body and hand positions) for sign language recognition
- Makes the learning process more efficient and potentially more accurate
- Provides a more consistent representation across different lighting conditions and backgrounds

### 2.3: Data Splitting

#### Overview
This module prepares the extracted keypoint data for model training by cleaning the data, organizing it by label, and splitting it into training, validation, and test sets. This step is crucial for ensuring that the model can be properly trained and evaluated.

#### Dependencies
- os: For file system operations
- random: For shuffling data during splitting
- json: For reading and writing structured data

#### Configuration
```python
KEYPOINTS_PATH = "../dataset/keypoints"  # Path to the keypoints JSON files
OUTPUT_DIR = "../dataset/data"           # Directory to save the split datasets
```

#### Functions

##### `create_dir(directory_path)`
Creates a directory if it doesn't already exist.

**Parameters:**
- `directory_path` (str): Path to the directory to be created

##### `load_keypoints_with_label(keypoints_path, label)`
Loads keypoints data for a specific label, filters out invalid entries, and adds the label information to each data point.

**Parameters:**
- `keypoints_path` (str): Path to the folder containing JSON files for keypoints
- `label` (str): The label identifier (e.g., '1', 'A')

**Returns:**
- List of cleaned keypoints data with label information included

**Data Cleaning Process:**
- Removes entries where all keypoints are zero (invalid detections)
- Adds the label information to each data point for model training

##### `split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)`
Splits the data into training, validation, and test sets according to the specified ratios.

**Parameters:**
- `data` (list): List of keypoints data
- `train_ratio` (float): Proportion of data for training (default: 0.8 or 80%)
- `val_ratio` (float): Proportion of data for validation (default: 0.1 or 10%)
- `test_ratio` (float): Proportion of data for testing (default: 0.1 or 10%)

**Returns:**
- Tuple containing three lists: (train_data, val_data, test_data)

**Process:**
1. Shuffles the data randomly to ensure unbiased splitting
2. Calculates index positions based on the specified ratios
3. Divides the data into the three subsets

##### `save_split_data(train_data, val_data, test_data, output_path)`
Saves the split data into separate JSON files for training, validation, and testing.

**Parameters:**
- `train_data` (list): List of keypoints data for training
- `val_data` (list): List of keypoints data for validation
- `test_data` (list): List of keypoints data for testing
- `output_path` (str): Path to save the split JSON files

#### Main Process Flow
1. Creates the output directory if it doesn't exist
2. Initializes empty lists for training, validation, and test data
3. Processes each label (1-9, A-Z):
   - Loads and cleans the keypoints data for the label
   - Splits the data into training, validation, and test subsets
   - Adds the split data to the respective global lists
4. Saves the combined data for all labels into three separate JSON files:
   - `train_data.json`: Contains all training data across all labels
   - `val_data.json`: Contains all validation data across all labels
   - `test_data.json`: Contains all test data across all labels

#### Data Structure
Each entry in the JSON files follows this structure:
```json
{
  "label": "A",
  "image_name": "filename.jpg",
  "keypoints": {
    "body": [
      {"x": 0.45, "y": 0.32, "z": 0.01},
      ...
    ],
    "hands": [
      {"x": 0.56, "y": 0.25, "z": 0.02},
      ...
    ]
  }
}
```

#### Output
The module creates three JSON files in the `../dataset/data` directory:
- `train_data.json`: Contains approximately 80% of the data for model training
- `val_data.json`: Contains approximately 10% of the data for model validation during training
- `test_data.json`: Contains approximately 10% of the data for final model evaluation

#### Purpose of Data Splitting
- **Training data**: Used to train the model's parameters
- **Validation data**: Used to tune hyperparameters and prevent overfitting during training
- **Test data**: Used to evaluate the final model's performance on unseen data

### 2.4: Feature Extraction and Normalization

#### Overview
This module processes the split data files (train, validation, test) by extracting keypoints, normalizing them into a consistent format, removing invalid samples, balancing class distribution, and encoding labels for model training. The output is a set of NumPy arrays ready for direct input into deep learning models.

#### Dependencies
- os: For file system operations
- numpy: For numerical array operations
- json: For reading structured data
- sklearn.model_selection: For train_test_split functionality
- sklearn.preprocessing.LabelEncoder: For converting text labels to numerical values
- tensorflow.keras.utils.to_categorical: For one-hot encoding of labels
- collections.defaultdict: For organizing data by class
- random: For balanced sampling

#### Constants
```python
BODY_KEYPOINTS = 33 * 3  # (x, y, z) for each body keypoint
HAND_KEYPOINTS = 42 * 3  # (x, y, z) for each hand keypoint (21 per hand)
MAX_LEN = BODY_KEYPOINTS + HAND_KEYPOINTS  # Total feature vector length
```

#### Functions

##### `load_split_data(json_file)`
Loads pre-split data from a JSON file.

**Parameters:**
- `json_file` (str): Path to the JSON file containing keypoints data

**Returns:**
- JSON data loaded from the file

**Error Handling:**
- Raises `FileNotFoundError` if the specified file doesn't exist

##### `normalize_keypoints(body, hands)`
Transforms structured keypoint data into a flat numerical array of fixed length.

**Parameters:**
- `body` (list): List of body keypoints, each containing x, y, z coordinates
- `hands` (list): List of hand keypoints, each containing x, y, z coordinates

**Returns:**
- numpy.ndarray: Flattened array of keypoints with consistent dimensions

**Process:**
1. Extracts body keypoints and converts them to a flat list of [x, y, z] values
2. Handles missing body keypoints by filling with zeros
3. Extracts hand keypoints and converts them similarly
4. Handles missing hand keypoints by filling with zeros
5. Ensures the output array has exactly `MAX_LEN` elements by:
   - Padding with zeros if shorter than required
   - Trimming if longer than required

##### `remove_invalid_data(json_data)`
Filters out data samples where hand keypoints are listed but contain only zeros.

**Parameters:**
- `json_data` (list): List of data samples containing keypoints

**Returns:**
- list: Filtered data with invalid samples removed

**Invalid Sample Definition:**
- Samples where the hands list is non-empty but all x, y, z values are zero (indicating failed hand tracking)

##### `balance_data(data)`
Balances the dataset by downsampling majority classes to match the size of the smallest class.

**Parameters:**
- `data` (list): List of data samples, each containing a label

**Returns:**
- list: Balanced dataset with equal representation across classes

**Process:**
1. Organizes samples by their class labels
2. Determines the smallest class size
3. For classes with more samples than the minimum:
   - Randomly samples (without replacement) to reduce the class size
4. Combines all downsampled classes into a balanced dataset

##### `prepare_data(json_data, label_encoder)`
Prepares keypoint data and corresponding labels for model training.

**Parameters:**
- `json_data` (list): List of data samples containing keypoints and labels
- `label_encoder` (LabelEncoder): Encoder for converting text labels to numerical values

**Returns:**
- tuple: (X, y) where X contains feature vectors and y contains encoded labels

**Process:**
1. For each data sample:
   - Normalizes keypoints into a feature vector
   - Extracts the corresponding label
2. Converts labels to numerical values using the label encoder
3. Returns both features and labels as NumPy arrays

##### `split_data(train_json, val_json, test_json)`
Processes the split data files into normalized feature vectors and encoded labels.

**Parameters:**
- `train_json` (str): Path to the training data JSON file
- `val_json` (str): Path to the validation data JSON file
- `test_json` (str): Path to the test data JSON file

**Returns:**
- tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder)

**Process:**
1. Loads data from each JSON file
2. Removes invalid samples from all datasets
3. Balances the training data to prevent class bias
4. Creates and fits a label encoder using training labels
5. Prepares feature vectors and labels for all datasets
6. Converts labels to one-hot encoded format
7. Returns processed datasets and the label encoder

#### Main Process Flow
1. Loads the split JSON files (train, validation, test)
2. Processes all datasets through the defined functions
3. Saves the processed data as NumPy arrays:
   - `dataset.npz`: Contains all processed feature and label arrays
   - `label_encoder.npy`: Contains the class names for interpreting predictions
4. Prints diagnostic information about the datasets

#### Output
The module creates two files:
- `dataset.npz`: A compressed NumPy archive containing:
  - `X_train`, `y_train`: Training features and labels
  - `X_val`, `y_val`: Validation features and labels
  - `X_test`, `y_test`: Test features and labels
- `label_encoder.npy`: The label encoder's classes array for mapping numerical predictions back to sign language symbols

#### Diagnostic Information
The script outputs:
- Number of samples in each dataset split
- Sample information for the first 5 training samples:
  - Number of body keypoints
  - Number of hand keypoints
  - Total keypoint count
- Count of samples with missing keypoints

#### Purpose
This step transforms raw keypoint data into a standardized format that:
1. Has consistent dimensions across all samples
2. Represents each sign language symbol equally in the training data
3. Separates features (keypoints) from targets (labels)
4. Uses numerical encoding suitable for deep learning models
5. Maintains separate datasets for training, validation, and testing

## Step 3: Model Training

### Overview
This module implements a sophisticated neural network architecture that combines Bidirectional LSTMs, CNNs, and an attention mechanism. The model is designed for sequence classification tasks and follows a hybrid architecture pattern to leverage the strengths of both recurrent and convolutional layers for sign language recognition.

### Dependencies
- TensorFlow/Keras: For building and training the neural network
- NumPy: For numerical array operations
- Matplotlib: For visualizing training progress and results
- Scikit-learn: For evaluation metrics
- Seaborn: For creating visualization heatmaps

### Model Architecture

#### Core Components

1. **Input Layer**
   - Accepts time series data with shape: (sequence_length, 1)
   - Processes the normalized keypoint data from the preprocessing step

2. **Recurrent Neural Network Blocks**
   - Three stacked Bidirectional LSTM layers (256, 128, 64 units)
   - One final standard LSTM layer (32 units)
   - Return sequences enabled to preserve temporal information

3. **Convolutional Neural Network Blocks**
   - Three CNN blocks interleaved with the LSTM layers
   - Decreasing filter sizes: 256 → 128 → 64
   - Kernel size of 3 for all convolutional layers
   - MaxPooling after each convolution (pool size = 2)
   - Batch normalization for training stability

4. **Attention Mechanism**
   - Custom attention block applied to the final LSTM output
   - Uses soft attention to focus on relevant time steps
   - Implemented through trainable weights and softmax activation

5. **Dense Layers**
   - 128-unit dense layer with ReLU activation
   - Final output layer with softmax activation for classification

#### Regularization Techniques

- **Dropout:** Applied after each LSTM layer (0.4) and before the output layer (0.3)
- **L2 Regularization:** Applied to all LSTM and CNN layers (lambda = 0.0005)
- **Batch Normalization:** Applied after each CNN block

### Training Configuration

- **Optimizer:** Adam with initial learning rate of 0.0002
- **Loss Function:** Categorical Cross-Entropy
- **Batch Size:** 128
- **Maximum Epochs:** 100
- **Callbacks:**
  - Model checkpoint to save the best model based on validation accuracy
  - Early stopping with patience of 10 epochs
  - Learning rate reduction on plateau (factor=0.5, patience=3)

### Custom Functions

#### Attention Mechanism

```python
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    # Learn an attention vector of size (time_steps, 1)
    a = Dense(1, activation='tanh')(inputs)
    a = Flatten()(a)
    a = Activation('softmax')(a)
    a = RepeatVector(input_dim)(a)
    a = Permute([2, 1])(a)
    # Apply the attention weights
    output = Multiply()([inputs, a])
    # Sum over time steps to get a context vector
    output = Lambda(sum_over_time)(output)
    return output
```

The attention mechanism:
1. Computes an attention vector for each time step
2. Normalizes attention weights using softmax
3. Multiplies the original sequence by attention weights
4. Sums over time steps to create a context vector

#### Helper Functions

```python
def sum_over_time(x):
    return tf.keras.backend.sum(x, axis=1)
```

### Evaluation Metrics

- **Accuracy:** Primary metric for model performance
- **Loss:** Categorical cross-entropy loss
- **Confusion Matrix:** Visual representation of classification performance
- **Classification Report:** Precision, recall, and F1-score for each class

### Performance Visualization

- Training and validation accuracy over epochs
- Training and validation loss over epochs
- Confusion matrix heatmap

### Model Summary

The model contains:
- Multiple Bidirectional LSTM layers
- Convolutional layers with max pooling
- Custom attention mechanism
- Dropout and L2 regularization for preventing overfitting
- A total of approximately 1-2 million parameters (exact count depends on input dimensions)

### Usage

1. **Loading the model:**
   ```python
   from tensorflow.keras.models import load_model
   model = load_model("model/best_model.keras", 
                      custom_objects={"sum_over_time": sum_over_time})
   ```

2. **Making predictions:**
   ```python
   # Ensure input data has shape (samples, sequence_length, 1)
   predictions = model.predict(X_test)
   predicted_classes = np.argmax(predictions, axis=1)
   ```

### Notes on Architecture Design

This hybrid architecture combines the strengths of different neural network types:

- **Bidirectional LSTMs:** Capture temporal patterns from both past and future context
- **CNNs:** Extract local patterns and features from sequences
- **Attention:** Focuses on the most informative parts of the sequence
- **Interleaved approach:** Allows for hierarchical feature extraction

This design is particularly effective for sequence classification tasks where both local features and global context are important, making it well-suited for sign language recognition based on keypoint data.

## Step 4: Application Implementation

### Overview
The project includes a graphical user interface (GUI) application that provides real-time sign language recognition using a webcam. The application allows users to form words and sentences through sign language gestures and even perform web searches based on the recognized text.

### Dependencies
- tkinter: For building the GUI interface
- OpenCV: For webcam capture and image processing
- MediaPipe: For real-time pose and hand landmark detection
- TensorFlow/Keras: For loading the trained model and making predictions
- autocorrect: For spelling correction of recognized words

### Features
- **Real-time Recognition**: Detects and interprets sign language gestures in real-time
- **Word Formation**: Accumulates recognized letters to form words
- **Sentence Building**: Automatically adds spaces between words when no gestures are detected for a period
- **Spelling Suggestions**: Provides spelling corrections for recognized words
- **Web Search Integration**: Allows searching the web using recognized text
- **User-friendly Interface**: Clean, intuitive interface with camera controls and text display

### Implementation Details

#### Prediction Pipeline
1. Capture video frames from webcam
2. Extract body and hand keypoints using MediaPipe
3. Normalize keypoints to match training data format
4. Feed normalized keypoints to the trained model
5. Apply temporal smoothing using a prediction buffer
6. Convert predictions to letters, words, and sentences
7. Apply spelling correction to improve accuracy

#### User Interface
- Camera toggle button for starting/stopping webcam
- Real-time prediction display
- Current word and spelling suggestions display
- Complete sentence display
- Search functionality for web queries
- Keyboard shortcuts for common actions

#### Performance Optimization
- Multi-threading for non-blocking UI during video processing
- Temporal smoothing to reduce prediction jitter
- Frame rate optimization for real-time performance

### Usage
1. Run the application: `python predict/app.py`
2. Press 'c' to start the camera
3. Make sign language gestures to form words
4. Wait briefly between words to add spaces
5. Use the search functionality to search the web with recognized text
6. Press 'q' to quit the application