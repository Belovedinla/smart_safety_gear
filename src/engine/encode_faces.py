import os
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.detection.face_recognizer import FaceRecognizer

def main():
    print("Initializing face recognizer...")
    recognizer = FaceRecognizer()
    
    faces_dir = "data/employee_faces"
    if not os.path.exists(faces_dir) or not os.listdir(faces_dir):
        print(f"ERROR: No images found in {faces_dir}.")
        print("Please add some photos of employees (e.g., 'John_Doe.jpg') to the folder and try again.")
        return
        
    recognizer.encode_faces(faces_dir=faces_dir)
    print("Done!")

if __name__ == "__main__":
    main()
