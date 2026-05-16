import streamlit as st
import cv2
import tempfile
import os
import pandas as pd

from src.detection.yolo_detector import YoloDetector
from src.detection.face_recognizer import FaceRecognizer
from src.database.db_manager import DatabaseManager
from src.engine.violation_engine import ViolationEngine

st.set_page_config(page_title="Smart Safety Gear Detection", layout="wide")

@st.cache_resource
def load_components():
    db = DatabaseManager()
    yolo = YoloDetector()
    face_rec = FaceRecognizer()
    engine = ViolationEngine(db)
    return db, yolo, face_rec, engine

db, yolo, face_rec, engine = load_components()

st.title("Smart Safety Gear Detection System")

menu = ["Dashboard", "Live Camera", "Upload Media", "Logs"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Dashboard":
    st.subheader("Dashboard Overview")
    col1, col2 = st.columns(2)
    todays_violations = db.get_todays_stats()
    col1.metric("Today's Violations", todays_violations)
    col2.metric("System Status", "Online")
    
    st.info("Navigate via the sidebar to access live monitoring or upload videos.")

elif choice == "Upload Media":
    st.subheader("Upload Image or Video for Inspection")
    uploaded_file = st.file_uploader("Upload Media", type=['jpg', 'png', 'jpeg', 'mp4'])
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1]
        
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.'+file_ext)
        tfile.write(uploaded_file.read())
        
        if file_ext in ['jpg', 'jpeg', 'png']:
            img = cv2.imread(tfile.name)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            detections = yolo.detect(img)
            bboxes = yolo.get_bboxes(detections)
            face_locations, face_names = face_rec.recognize(img_rgb)
            violations = engine.process_detections(img, bboxes, face_locations, face_names)
            
            annotated_frame = engine.draw_annotations(img, bboxes, violations, face_locations, face_names)
            
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            
        elif file_ext == 'mp4':
            stframe = st.empty()
            cap = cv2.VideoCapture(tfile.name)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = yolo.detect(frame)
                bboxes = yolo.get_bboxes(detections)
                
                face_locations, face_names = face_rec.recognize(frame_rgb)
                violations = engine.process_detections(frame, bboxes, face_locations, face_names)
                annotated_frame = engine.draw_annotations(frame, bboxes, violations, face_locations, face_names)
                
                stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            cap.release()

elif choice == "Live Camera":
    st.subheader("Live Monitoring")
    st.warning("Webcam access requires https or localhost.")
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])
    
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture from webcam.")
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = yolo.detect(frame)
        bboxes = yolo.get_bboxes(detections)
        face_locations, face_names = face_rec.recognize(frame_rgb)
        violations = engine.process_detections(frame, bboxes, face_locations, face_names)
        annotated_frame = engine.draw_annotations(frame, bboxes, violations, face_locations, face_names)
        
        FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    else:
        cap.release()

elif choice == "Logs":
    st.subheader("Violation Logs")
    records = db.get_recent_violations(50)
    if records:
        df = pd.DataFrame(records, columns=["ID", "Timestamp", "Employee", "Missing PPE", "Risk Level", "Snapshot"])
        st.dataframe(df)
    else:
        st.info("No violations logged yet.")
