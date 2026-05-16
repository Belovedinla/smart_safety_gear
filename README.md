# Smart Safety Gear Detection System

An end-to-end AI/ML application for workplace safety monitoring. It detects Personal Protective Equipment (PPE) using YOLOv8, recognizes employees using CNN-based face recognition, and logs safety violations to a database.

## Features
- **PPE Detection:** Helmet, Vest, Gloves, Mask, Goggles using YOLOv8.
- **Face Recognition:** Identifies employees for accountability.
- **Violation Engine:** Cross-references PPE with identified persons.
- **Streamlit Dashboard:** Real-time monitoring, video upload, and logs.

## Setup
1. Clone the repository.
2. `python -m venv venv`
3. Activate the environment.
4. `pip install -r requirements.txt`
5. Run the app: `streamlit run app.py`

## Architecture
(See implementation plan for details)
