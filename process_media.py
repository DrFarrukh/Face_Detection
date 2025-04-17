import cv2
import torch
import numpy as np
import time
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from process_known_faces import process_known_faces
from collections import deque
import argparse

class FastMTCNN(object):
    """Fast MTCNN implementation for real-time face detection."""
    
    def __init__(self, stride=4, resize=1.0, *args, **kwargs):
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        self.frames_buffer = deque(maxlen=stride)
        self.last_boxes = None
        
    def __call__(self, frame):
        if self.resize != 1:
            frame = cv2.resize(frame, 
                (int(frame.shape[1] * self.resize), 
                 int(frame.shape[0] * self.resize)))
        
        self.frames_buffer.append(frame)
        
        if len(self.frames_buffer) == self.stride:
            try:
                batch_boxes, probs, _ = self.mtcnn.detect(frame, landmarks=True)
                if batch_boxes is not None:
                    self.last_boxes = batch_boxes
            except Exception as e:
                print(f"Detection error: {e}")
            
            self.frames_buffer.clear()
        
        return self.last_boxes if self.last_boxes is not None else None

def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def process_frame(frame, fast_mtcnn, resnet, known_embeddings):
    """Process a single frame for face detection and recognition."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        batch_boxes = fast_mtcnn(rgb_frame)
        
        if batch_boxes is not None:
            faces = fast_mtcnn.mtcnn(rgb_frame)
            
            if faces is not None:
                if not isinstance(faces, list):
                    faces = [faces]
                
                for i, (box, face) in enumerate(zip(batch_boxes, faces)):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    
                    if face.dim() == 3:
                        face = face.unsqueeze(0)
                    
                    face_embedding = resnet(face)
                    face_embedding_np = face_embedding.detach().numpy()

                    max_similarity = 0
                    recognized_name = "Unknown"
                    
                    for name, known_embedding in known_embeddings.items():
                        similarity = cosine_similarity(face_embedding_np[0], known_embedding[0])
                        if similarity > max_similarity and similarity > 0.7:
                            max_similarity = similarity
                            recognized_name = name

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{recognized_name} ({max_similarity:.2f})", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    except Exception as e:
        print(f"Error processing frame: {e}")
    
    return frame

def process_video(input_path, output_path):
    """Process a video file for face detection and recognition."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fast_mtcnn = FastMTCNN(
        stride=3,
        resize=1.0,
        margin=14,
        factor=0.6,
        keep_all=True,
        device=device,
        post_process=True
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    known_embeddings = process_known_faces()
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            current_fps = frame_count / (time.time() - start_time)
            print(f"Processing frame {frame_count}, FPS: {current_fps:.1f}")
        
        # Process frame
        processed_frame = process_frame(frame, fast_mtcnn, resnet, known_embeddings)
        out.write(processed_frame)
    
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to {output_path}")

def process_image(input_path, output_path):
    """Process a single image for face detection and recognition."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fast_mtcnn = FastMTCNN(
        stride=1,  # No need for stride with single image
        resize=1.0,
        margin=14,
        factor=0.6,
        keep_all=True,
        device=device,
        post_process=True
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    known_embeddings = process_known_faces()
    
    # Read image
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error: Could not read image file {input_path}")
        return
    
    # Process image
    processed_frame = process_frame(frame, fast_mtcnn, resnet, known_embeddings)
    
    # Save processed image
    cv2.imwrite(output_path, processed_frame)
    print(f"Image processing complete. Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process video/image file for face recognition')
    parser.add_argument('input_path', help='Path to input video or image file')
    parser.add_argument('output_path', help='Path to save the output file')
    parser.add_argument('--type', choices=['video', 'image'], help='Type of input file (video or image)')
    
    args = parser.parse_args()
    
    # Determine input type if not specified
    if not args.type:
        _, ext = os.path.splitext(args.input_path.lower())
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            args.type = 'video'
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            args.type = 'image'
        else:
            print("Error: Could not determine input file type. Please specify --type")
            return
    
    # Process according to input type
    if args.type == 'video':
        process_video(args.input_path, args.output_path)
    else:
        process_image(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
