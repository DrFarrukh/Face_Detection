import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from process_known_faces import process_known_faces

def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def main():
    # Initialize MTCNN and InceptionResnetV1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # Load known face embeddings
    known_embeddings = process_known_faces()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect faces and get aligned face tensors
            batch_boxes, probs, _ = mtcnn.detect(rgb_frame, landmarks=True)
            
            if batch_boxes is not None:
                # Get aligned faces
                faces = mtcnn(rgb_frame)
                
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

        # Display the frame
        cv2.imshow('Face Recognition', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
