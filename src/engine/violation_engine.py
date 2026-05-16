import cv2

class ViolationEngine:
    def __init__(self, db_manager):
        # We require hardhat and vest based on the dataset classes
        self.required_ppe = ['hardhat', 'vest'] 
        self.db = db_manager

    def check_overlap(self, boxA, boxB):
        """Check if boxA and boxB overlap or are close to each other vertically."""
        # Simple overlap check, expanding the face box to represent a 'body'
        face_x1, face_y1, face_x2, face_y2 = boxB
        
        # Estimate body box based on face
        face_height = face_y2 - face_y1
        face_width = face_x2 - face_x1
        
        body_x1 = max(0, face_x1 - face_width)
        body_x2 = face_x2 + face_width
        body_y1 = max(0, face_y1 - int(face_height * 0.5)) # hardhat can be above
        body_y2 = face_y2 + face_height * 6 # vest is below
        
        # Check if boxA center is within the estimated body region
        cx = (boxA[0] + boxA[2]) / 2
        cy = (boxA[1] + boxA[3]) / 2
        
        if body_x1 <= cx <= body_x2 and body_y1 <= cy <= body_y2:
            return True
        return False

    def process_detections(self, frame, detections, faces, face_names):
        # In this dataset, there is no "Person" class. 
        # We will use the detected Faces as our "People" base.
        
        violations = []

        for (top, right, bottom, left), person_name in zip(faces, face_names):
            face_box = [left, top, right, bottom]
            
            # Find PPE associated with this person (face)
            person_ppe = []
            for d in detections:
                if d['class_name'] in self.required_ppe or d['class_name'] in ['glasses', 'gloves']:
                    if self.check_overlap(d['bbox'], face_box):
                        person_ppe.append(d['class_name'])
            
            missing = [req for req in self.required_ppe if req not in person_ppe]
            
            if missing:
                missing_str = ", ".join(missing)
                risk = "High" if len(missing) >= 2 else "Medium"
                violations.append({
                    "name": person_name,
                    "missing": missing_str,
                    "risk": risk,
                    "bbox": face_box # using face box for drawing the warning
                })
                # Log to DB
                self.db.log_violation(person_name, missing_str, risk)

        return violations

    def draw_annotations(self, frame, detections, violations, faces, face_names):
        annotated_frame = frame.copy()
        
        # Draw YOLO PPE Detections
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            color = (0, 255, 0) # Green for PPE
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, d['class_name'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw Face/Person Warnings
        for (top, right, bottom, left), name in zip(faces, face_names):
            face_box = [left, top, right, bottom]
            
            person_violation = next((v for v in violations if v['bbox'] == face_box), None)
            
            if person_violation:
                color = (0, 0, 255) # Red for violation
                text = f"{name} - Missing: {person_violation['missing']}"
            else:
                color = (255, 255, 0) # Cyan for safe person
                text = f"{name} - Safe"

            cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
            # Put text above the face
            cv2.putText(annotated_frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        return annotated_frame
