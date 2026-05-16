import os
import pickle
import cv2
import face_recognition

class FaceRecognizer:
    def __init__(self, encodings_path="data/encodings/encodings.pkl", tolerance=0.5):
        self.encodings_path = encodings_path
        self.tolerance = tolerance
        self.known_encodings = []
        self.known_names = []
        self.load_encodings()

    def encode_faces(self, faces_dir="data/employee_faces"):
        if not os.path.exists(faces_dir):
            print(f"Directory {faces_dir} not found.")
            return

        print("Generating face encodings...")
        for filename in os.listdir(faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0].replace('_', ' ').title()
                filepath = os.path.join(faces_dir, filename)
                
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.known_encodings.append(encodings[0])
                    self.known_names.append(name)
                    print(f"Encoded face for: {name}")
                else:
                    print(f"No face found in {filename}")
        
        os.makedirs(os.path.dirname(self.encodings_path), exist_ok=True)
        with open(self.encodings_path, 'wb') as f:
            pickle.dump({"encodings": self.known_encodings, "names": self.known_names}, f)
        print(f"Encodings saved to {self.encodings_path}")

    def load_encodings(self):
        if os.path.exists(self.encodings_path):
            with open(self.encodings_path, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
            print(f"Loaded {len(self.known_names)} encodings.")
        else:
            print("No encodings found. Please run encode_faces() first.")

    def recognize(self, rgb_image, face_locations=None):
        if not self.known_encodings:
            return face_locations or [], []
            
        if face_locations is None:
            face_locations = face_recognition.face_locations(rgb_image)
        
        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        names = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(self.known_encodings, encoding, tolerance=self.tolerance)
            name = "Unknown"

            if True in matches:
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matched_idxs:
                    n = self.known_names[i]
                    counts[n] = counts.get(n, 0) + 1
                name = max(counts, key=counts.get)
            
            names.append(name)

        return face_locations, names
