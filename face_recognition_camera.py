import cv2
import torch
import numpy as np
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
from process_known_faces import process_known_faces
from collections import deque

class FastMTCNN(object):
    """Fast MTCNN implementation for real-time face detection."""
    
    def __init__(self, stride=4, resize=1.0, *args, **kwargs):
        """Initialize the FastMTCNN object.
        
        Args:
            stride (int): Detection stride. Faces are detected every `stride` frames.
            resize (float): Frame scaling factor.
            *args, **kwargs: Arguments passed to MTCNN constructor.
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        self.frames_buffer = deque(maxlen=stride)
        self.last_boxes = None
        
    def __call__(self, frame):
        """Detect faces in frame using strided MTCNN."""
        if self.resize != 1:
            frame = cv2.resize(frame, 
                (int(frame.shape[1] * self.resize), 
                 int(frame.shape[0] * self.resize)))
        
        self.frames_buffer.append(frame)
        
        # Only run detection every 'stride' frames
        if len(self.frames_buffer) == self.stride:
            try:
                batch_boxes, probs, _ = self.mtcnn.detect(frame, landmarks=True)
                if batch_boxes is not None:
                    self.last_boxes = batch_boxes
            except Exception as e:
                print(f"Detection error: {e}")
            
            self.frames_buffer.clear()
        
        # Return the last known boxes
        return self.last_boxes if self.last_boxes is not None else None

def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def main():
    # Initialize FastMTCNN and InceptionResnetV1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fast_mtcnn = FastMTCNN(
        stride=10,  # Detect faces every 3 frames
        resize=1.0,  # Use full resolution
        margin=14,
        factor=0.6,  # Smaller scaling factor for better performance
        keep_all=True,
        device=device,
        post_process=True
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # Load known face embeddings
    known_embeddings = process_known_faces()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:  # Update FPS every 30 frames
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = time.time()

        # Convert frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect faces using FastMTCNN
            batch_boxes = fast_mtcnn(rgb_frame)
            
            if batch_boxes is not None:
                # Get aligned faces
                faces = fast_mtcnn.mtcnn(rgb_frame)
                
                if faces is not None:
                    # Handle both single and multiple face cases
                    if not isinstance(faces, list):
                        faces = [faces]
                    
                    for i, (box, face) in enumerate(zip(batch_boxes, faces)):
                        # Get coordinates for face bounding box
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        
                        # Ensure face tensor has correct dimensions
                        if face.dim() == 3:
                            face = face.unsqueeze(0)
                        
                        # Get face embedding
                        face_embedding = resnet(face)
                        face_embedding_np = face_embedding.detach().numpy()

                        # Compare with known faces
                        max_similarity = 0
                        recognized_name = "Unknown"
                        
                        for name, known_embedding in known_embeddings.items():
                            similarity = cosine_similarity(face_embedding_np[0], known_embedding[0])
                            if similarity > max_similarity and similarity > 0.7:  # Threshold for recognition
                                max_similarity = similarity
                                recognized_name = name

                        # Draw bounding box and name
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{recognized_name} ({max_similarity:.2f})", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        # Display FPS in the top-left corner
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Face Recognition', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
