# Real-time Face Recognition System

This project implements a real-time face recognition system using Python, OpenCV, and FaceNet. The system can detect and recognize faces from a USB camera feed by comparing them with known face images stored in a directory.

## Features

- Real-time face detection using MTCNN (Multi-task Cascaded Convolutional Networks)
- Face recognition using FaceNet's InceptionResnetV1 model
- Support for multiple face detection and recognition simultaneously
- Easy addition of new faces to recognize
- GPU acceleration support (if available)
- Simple and intuitive user interface

## Requirements

- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)

### Required Python Packages

```
opencv-python==4.8.1.78
tensorflow==2.13.0
numpy==1.24.3
facenet-pytorch==2.5.3
mtcnn==0.1.1
```

## Project Structure

```
Face_Detection/
│
├── face_recognition_camera.py  # Main script for real-time recognition
├── process_known_faces.py      # Script to process known face images
├── requirements.txt            # Python dependencies
│
├── known_faces/               # Directory for storing reference face images
│   └── (your images).jpg      # Named as person_name.jpg
│
└── models/                    # Directory for storing model weights (auto-downloaded)
```

## Installation

1. Clone or download this repository
2. Install Python 3.7 or higher
3. Install required packages:
   ```powershell
   python -m pip install -r requirements.txt
   ```

## Usage

### 1. Adding Known Faces

1. Create clear, well-lit photos of the people you want to recognize
2. Save these photos in the `known_faces` directory
3. Name each photo as `person_name.jpg` (e.g., `john.jpg`, `anna.jpg`)
   - Supported formats: .jpg, .jpeg, .png
   - One face per image recommended
   - Face should be clearly visible and well-lit

### 2. Running the Recognition System

1. Make sure your USB camera is connected
2. Run the main script:
   ```powershell
   python face_recognition_camera.py
   ```
3. The system will:
   - Load known face images and generate embeddings
   - Open your webcam feed
   - Show detected faces with green bounding boxes
   - Display recognized names above the faces

### Controls

- Press 'q' to quit the application
- Recognition threshold is set to 0.7 (can be adjusted in the code for more/less strict matching)

## How It Works

1. **Face Detection**: Uses MTCNN to detect faces in each frame from the camera
2. **Face Embedding**: Uses FaceNet (InceptionResnetV1) to generate embeddings for detected faces
3. **Recognition**: Compares embeddings with known face embeddings using cosine similarity
4. **Display**: Shows results in real-time with bounding boxes and names

## Performance Notes

- GPU acceleration is automatically used if available
- Recognition accuracy depends on:
  - Quality of reference photos
  - Lighting conditions
  - Face angle and expression
  - Distance from camera

## Troubleshooting

1. **Camera not detected**: 
   - Ensure your USB camera is properly connected
   - Try changing `cv2.VideoCapture(0)` to a different number if you have multiple cameras

2. **Poor recognition**:
   - Improve lighting conditions
   - Update reference photos
   - Adjust the similarity threshold (default: 0.7)

3. **Slow performance**:
   - Consider using GPU acceleration
   - Reduce frame resolution if needed
   - Ensure proper lighting to help with detection

## License

This project is available for educational and personal use.

## Dependencies Credit

- FaceNet-PyTorch: https://github.com/timesler/facenet-pytorch
- MTCNN: https://github.com/timesler/facenet-pytorch
- OpenCV: https://opencv.org/
