import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np

def process_known_faces():
    # Initialize MTCNN for face detection and InceptionResnetV1 for face recognition
    mtcnn = MTCNN(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    known_embeddings = {}
    known_faces_dir = 'known_faces'

    # Process each image in the known_faces directory
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Get person name from filename (assuming format: name.jpg)
            person_name = os.path.splitext(filename)[0]
            
            # Read and process image
            img_path = os.path.join(known_faces_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect face and get embeddings
            face = mtcnn(img)
            if face is not None:
                face_embedding = resnet(face.unsqueeze(0))
                known_embeddings[person_name] = face_embedding.detach().numpy()
    
    return known_embeddings

if __name__ == '__main__':
    process_known_faces()
