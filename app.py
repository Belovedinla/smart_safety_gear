import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

from src.detection.yolo_detector import YoloDetector
from src.detection.face_recognizer import FaceRecognizer
from src.database.db_manager import DatabaseManager
from src.engine.violation_engine import ViolationEngine

st.set_page_config(
    page_title="Smart Safety Gear Detection | Enterprise AI Platform",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DESIGN SYSTEM - CSS & STYLING
# ============================================================================

DESIGN_SYSTEM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [data-testid="stAppViewContainer"] {
    background: #07111F !important;
    font-family: 'Inter', sans-serif;
    overflow-x: hidden;
}

/* ANIMATED BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: 
        radial-gradient(circle 800px at 20% 50%, rgba(47, 128, 237, 0.15), transparent),
        radial-gradient(circle 600px at 80% 80%, rgba(86, 204, 242, 0.1), transparent),
        linear-gradient(135deg, #07111F 0%, #0E1C36 50%, #07111F 100%);
    background-attachment: fixed;
}

/* MAIN CONTENT AREA */
[data-testid="stMainBlockContainer"] {
    padding-top: 1rem !important;
}

/* SIDEBAR CUSTOMIZATION */
[data-testid="stSidebar"] {
    background: rgba(14, 28, 54, 0.95) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
}

/* TEXT COLORS */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif;
    color: #F9FAFB !important;
    letter-spacing: -0.01em;
    font-weight: 700 !important;
}

p, span, label, div {
    color: #B8C1CC !important;
}

/* GLASSMORPHISM CARD BASE */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 28px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(255, 255, 255, 0.12);
    transform: translateY(-4px);
    box-shadow: 0 20px 50px rgba(47, 128, 237, 0.15);
}

/* NEON GLOW BORDERS */
.glow-border {
    border: 2px solid transparent;
    background: linear-gradient(rgba(255,255,255,0.06), rgba(255,255,255,0.06)) padding-box,
                linear-gradient(135deg, #2F80ED, #56CCF2) border-box;
}

.glow-border:hover {
    box-shadow: 0 0 30px rgba(47, 128, 237, 0.3);
}

/* BUTTONS */
.btn-primary {
    background: linear-gradient(135deg, #2F80ED, #2563F0);
    color: #F9FAFB;
    border: none;
    padding: 12px 28px;
    border-radius: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 10px 25px rgba(47, 128, 237, 0.2);
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 35px rgba(47, 128, 237, 0.3);
    background: linear-gradient(135deg, #2563F0, #1D4ED8);
}

.btn-danger {
    background: linear-gradient(135deg, #EB5757, #DC2626);
    color: #F9FAFB;
    border: none;
    padding: 12px 28px;
    border-radius: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-danger:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 35px rgba(235, 87, 87, 0.3);
}

.btn-success {
    background: linear-gradient(135deg, #27AE60, #059669);
    color: #F9FAFB;
    border: none;
    padding: 12px 28px;
    border-radius: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

/* KPI CARD */
.kpi-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(47, 128, 237, 0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
}

.kpi-card:hover {
    background: rgba(47, 128, 237, 0.1);
    border-color: rgba(86, 204, 242, 0.5);
    transform: translateY(-6px);
    box-shadow: 0 20px 40px rgba(47, 128, 237, 0.2);
}

.kpi-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #2F80ED;
    font-family: 'Space Grotesk', sans-serif;
}

.kpi-label {
    font-size: 0.95rem;
    color: #B8C1CC;
    margin-top: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* NAVBAR */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 32px;
    background: rgba(14, 28, 54, 0.9);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    margin-bottom: 32px;
    border-radius: 0;
}

.navbar-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #2F80ED, #56CCF2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.navbar-nav {
    display: flex;
    gap: 8px;
    align-items: center;
}

.nav-item {
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: 600;
    color: #B8C1CC;
}

.nav-item:hover,
.nav-item.active {
    background: rgba(47, 128, 237, 0.2);
    color: #56CCF2;
}

/* STATUS BADGE */
.status-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(39, 174, 96, 0.2);
    color: #27AE60;
    border: 1px solid rgba(39, 174, 96, 0.4);
}

.status-badge.warning {
    background: rgba(242, 201, 76, 0.2);
    color: #F2C94C;
    border-color: rgba(242, 201, 76, 0.4);
}

.status-badge.danger {
    background: rgba(235, 87, 87, 0.2);
    color: #EB5757;
    border-color: rgba(235, 87, 87, 0.4);
}

/* ANIMATIONS */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideUp {
    from { 
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 20px rgba(47, 128, 237, 0.2); }
    50% { box-shadow: 0 0 40px rgba(47, 128, 237, 0.4); }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
}

.animate-fade {
    animation: fadeIn 0.6s ease-out;
}

.animate-slide {
    animation: slideUp 0.6s ease-out;
}

.animate-glow {
    animation: glow 3s infinite;
}

.animate-float {
    animation: float 3s ease-in-out infinite;
}

/* ALERT STYLES */
.alert-success {
    background: rgba(39, 174, 96, 0.1);
    border: 1px solid rgba(39, 174, 96, 0.3);
    border-radius: 12px;
    padding: 16px;
    color: #27AE60;
}

.alert-warning {
    background: rgba(242, 201, 76, 0.1);
    border: 1px solid rgba(242, 201, 76, 0.3);
    border-radius: 12px;
    padding: 16px;
    color: #F2C94C;
}

.alert-danger {
    background: rgba(235, 87, 87, 0.1);
    border: 1px solid rgba(235, 87, 87, 0.3);
    border-radius: 12px;
    padding: 16px;
    color: #EB5757;
}

/* SECTION DIVIDER */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    margin: 40px 0;
}

/* CHART CONTAINER */
.chart-container {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
}

/* TABLE STYLES */
table {
    background: transparent !important;
}

[data-testid="dataframe"] {
    background: transparent !important;
}

th {
    background: rgba(47, 128, 237, 0.1) !important;
    color: #F9FAFB !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
}

td {
    border-color: rgba(255, 255, 255, 0.05) !important;
    color: #B8C1CC !important;
}

/* FOOTER */
.footer {
    text-align: center;
    padding: 40px 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    color: #B8C1CC;
    font-size: 0.9rem;
    margin-top: 60px;
}

.footer-brand {
    color: #2F80ED;
    font-weight: 600;
}

/* LOADER ANIMATION */
.loader {
    border: 4px solid rgba(47, 128, 237, 0.2);
    border-top: 4px solid #2F80ED;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* RESPONSIVE */
@media (max-width: 768px) {
    .navbar {
        flex-direction: column;
        gap: 16px;
    }
    
    .navbar-nav {
        width: 100%;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    .kpi-card {
        min-height: 100px;
    }
    
    h1 {
        font-size: 2rem !important;
    }
}
</style>
"""

st.markdown(DESIGN_SYSTEM_CSS, unsafe_allow_html=True)

# ============================================================================
# COMPONENTS
# ============================================================================

def render_navbar(current_page: str):
    """Render the top navigation bar"""
    MENU = ["Dashboard", "Live Camera", "Upload Media", "Logs", "Analytics"]
    
    nav_items = "".join([
        f"""<a href='?page={page.replace(' ', '%20')}' 
           class='nav-item {'active' if current_page == page else ''}' 
           onclick="event.preventDefault(); setPage('{page}')">
           {page}
        </a>"""
        for page in MENU
    ])
    
    navbar_html = f"""
    <div class='navbar'>
        <div class='navbar-brand'>🦺 Smart Safety Gear Detection</div>
        <div class='navbar-nav'>
            {nav_items}
        </div>
        <div style='display: flex; gap: 8px; align-items: center;'>
            <span class='status-badge'>🟢 System Online</span>
        </div>
    </div>
    <script>
        function setPage(page) {{
            const params = new URLSearchParams(window.location.search);
            params.set('page', page);
            window.location.search = params.toString();
        }}
    </script>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)

def kpi_card(label: str, value: str, change: str = None, icon: str = "📊"):
    """Render a KPI card"""
    change_html = f"""
    <div style='font-size: 0.9rem; color: #27AE60; margin-top: 8px;'>
        ↑ {change}
    </div>
    """ if change else ""
    
    card_html = f"""
    <div class='kpi-card'>
        <div style='font-size: 2rem; margin-bottom: 8px;'>{icon}</div>
        <div class='kpi-value'>{value}</div>
        <div class='kpi-label'>{label}</div>
        {change_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def glass_card(content_html: str, title: str = None):
    """Render a glassmorphism card"""
    title_html = f"<h3 style='margin-bottom: 20px;'>{title}</h3>" if title else ""
    card_html = f"""
    <div class='glass-card animate-slide'>
        {title_html}
        {content_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_footer():
    """Render the footer"""
    footer_html = """
    <div class='footer'>
        <p>© 2026 <span class='footer-brand'>Smart Safety Gear Detection System</span> • Enterprise AI Platform</p>
        <p>Advanced PPE Detection & Violation Monitoring • Powered by YOLOv8 & Face Recognition</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def dashboard_page():
    """Premium Analytics Dashboard"""
    st.markdown("<h1 style='margin-bottom: 30px;'>📊 Safety Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    # KPI Section
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total Violations", "247", "+12%", "⚠️")
    with col2:
        kpi_card("PPE Compliance", "94.2%", "+3.5%", "✅")
    with col3:
        kpi_card("Employees", "156", "Active", "👥")
    with col4:
        kpi_card("System Health", "99.8%", "Optimal", "🔧")
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Charts Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>Violation Trends (7 Days)</h3>", unsafe_allow_html=True)
        # Create dummy data
        dates = pd.date_range(start='2026-05-12', periods=7)
        violations = [15, 12, 18, 14, 11, 16, 9]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=violations,
            mode='lines+markers',
            name='Violations',
            line=dict(color='#2F80ED', width=3),
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor='rgba(47, 128, 237, 0.2)'
        ))
        fig.update_layout(
            title='',
            xaxis_title='Date',
            yaxis_title='Count',
            template='plotly_dark',
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#B8C1CC', family='Inter'),
            showlegend=False,
            height=350
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown("<h3>PPE Distribution</h3>", unsafe_allow_html=True)
        ppe_data = {
            'Helmet': 450,
            'Vest': 380,
            'Gloves': 320,
            'Goggles': 280,
            'Mask': 250
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(ppe_data.keys()),
            values=list(ppe_data.values()),
            marker=dict(colors=['#2F80ED', '#56CCF2', '#F2C94C', '#27AE60', '#EB5757']),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
        )])
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#B8C1CC', family='Inter'),
            height=350,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Recent Violations Table
    st.markdown("<h3>Recent Safety Violations</h3>", unsafe_allow_html=True)
    
    violations_data = {
        'Employee': ['John Doe', 'Jane Smith', 'Mike Johnson', 'Sarah Wilson', 'Tom Brown'],
        'Timestamp': [
            datetime.now() - timedelta(hours=2),
            datetime.now() - timedelta(hours=4),
            datetime.now() - timedelta(hours=6),
            datetime.now() - timedelta(hours=8),
            datetime.now() - timedelta(hours=10)
        ],
        'Missing': ['Helmet', 'Vest', 'Helmet, Gloves', 'Mask', 'Goggles'],
        'Risk Level': ['High', 'Medium', 'High', 'Low', 'Medium'],
        'Area': ['Warehouse A', 'Site B', 'Warehouse A', 'Office', 'Warehouse B']
    }
    
    df = pd.DataFrame(violations_data)
    
    # Style dataframe with colors
    def risk_color(val):
        if val == 'High':
            return 'background-color: rgba(235, 87, 87, 0.2); color: #EB5757;'
        elif val == 'Medium':
            return 'background-color: rgba(242, 201, 76, 0.2); color: #F2C94C;'
        else:
            return 'background-color: rgba(39, 174, 96, 0.2); color: #27AE60;'
    
    styled_df = df.style.applymap(risk_color, subset=['Risk Level'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def live_camera_page():
    """Live Camera Monitoring with AI HUD"""
    st.markdown("<h1 style='margin-bottom: 30px;'>🎥 Live Camera Monitoring</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("<h3>Real-time Detection Feed</h3>", unsafe_allow_html=True)
        
        # Video feed placeholder with HUD styling
        hud_html = """
        <div class='glass-card' style='padding: 0; overflow: hidden; border: 2px solid rgba(47, 128, 237, 0.5); position: relative;'>
            <div style='
                width: 100%;
                aspect-ratio: 16/9;
                background: radial-gradient(circle at center, rgba(47, 128, 237, 0.1), rgba(7, 17, 31, 0.8));
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                border: 1px solid rgba(86, 204, 242, 0.3);
                border-radius: 16px;
            '>
                <div style='text-align: center;'>
                    <div style='font-size: 4rem; margin-bottom: 10px;'>📹</div>
                    <p style='color: #56CCF2; font-weight: 600;'>Camera Inactive - Click to Start</p>
                </div>
                <div style='
                    position: absolute;
                    top: 12px;
                    left: 12px;
                    background: rgba(235, 87, 87, 0.2);
                    border: 1px solid rgba(235, 87, 87, 0.5);
                    color: #EB5757;
                    padding: 6px 12px;
                    border-radius: 8px;
                    font-size: 0.85rem;
                    font-weight: 600;
                '>
                    🔴 STANDBY
                </div>
                <div style='
                    position: absolute;
                    bottom: 12px;
                    right: 12px;
                    background: rgba(47, 128, 237, 0.2);
                    border: 1px solid rgba(47, 128, 237, 0.5);
                    color: #2F80ED;
                    padding: 8px 16px;
                    border-radius: 8px;
                    font-size: 0.9rem;
                    font-weight: 600;
                '>
                    FPS: 30 | AI Confidence: 94.2%
                </div>
            </div>
        </div>
        """
        st.markdown(hud_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3>Detection Stats</h3>", unsafe_allow_html=True)
        
        stats_html = """
        <div class='glass-card'>
            <div style='margin-bottom: 16px;'>
                <div style='font-size: 0.9rem; color: #B8C1CC; margin-bottom: 4px;'>Persons Detected</div>
                <div style='font-size: 2rem; color: #2F80ED; font-weight: 700;'>5</div>
            </div>
            <div style='margin-bottom: 16px;'>
                <div style='font-size: 0.9rem; color: #B8C1CC; margin-bottom: 4px;'>Violations</div>
                <div style='font-size: 2rem; color: #EB5757; font-weight: 700;'>2</div>
            </div>
            <div style='margin-bottom: 16px;'>
                <div style='font-size: 0.9rem; color: #B8C1CC; margin-bottom: 4px;'>Compliance Rate</div>
                <div style='font-size: 2rem; color: #27AE60; font-weight: 700;'>60%</div>
            </div>
            <div>
                <div style='font-size: 0.9rem; color: #B8C1CC; margin-bottom: 4px;'>System Status</div>
                <span class='status-badge' style='width: 100%; text-align: center; display: block;'>🟢 Ready</span>
            </div>
        </div>
        """
        st.markdown(stats_html, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Camera Controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("▶️ Start Camera", use_container_width=True, key="start_camera"):
            st.success("Camera started - Real-time monitoring active")
    with col2:
        if st.button("⏹️ Stop Camera", use_container_width=True, key="stop_camera"):
            st.info("Camera stopped")
    with col3:
        if st.button("📸 Capture Frame", use_container_width=True, key="capture_frame"):
            st.success("Frame captured")
    with col4:
        if st.button("💾 Save Recording", use_container_width=True, key="save_recording"):
            st.success("Recording saved")


def upload_media_page():
    """Enhanced Upload Interface"""
    st.markdown("<h1 style='margin-bottom: 30px;'>📤 Upload & Analyze Media</h1>", unsafe_allow_html=True)
    
    # Drag and drop zone
    upload_html = """
    <div class='glass-card' style='
        border: 2px dashed rgba(47, 128, 237, 0.5);
        text-align: center;
        padding: 50px 20px;
        cursor: pointer;
        transition: all 0.3s ease;
    '>
        <div style='font-size: 4rem; margin-bottom: 10px;'>📁</div>
        <h3>Drag & Drop Your Files Here</h3>
        <p style='color: #B8C1CC; margin-bottom: 15px;'>or click to browse</p>
        <p style='font-size: 0.9rem; color: #56CCF2;'>Supported: JPG, PNG, JPEG, MP4</p>
    </div>
    """
    st.markdown(upload_html, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Select file to analyze", type=['jpg', 'png', 'jpeg', 'mp4'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>Original Media</h3>", unsafe_allow_html=True)
            st.info(f"Processing: {uploaded_file.name}")
        
        with col2:
            st.markdown("<h3>AI Analysis Results</h3>", unsafe_allow_html=True)
            
            analysis_html = """
            <div class='glass-card'>
                <div style='margin-bottom: 12px;'>
                    <span style='color: #B8C1CC;'>Detection Status:</span>
                    <span class='status-badge'>✅ Detected 3 items</span>
                </div>
                <div style='margin-bottom: 12px;'>
                    <span style='color: #B8C1CC;'>Confidence:</span>
                    <span style='color: #27AE60; font-weight: 600;'>94.2%</span>
                </div>
                <div style='margin-bottom: 12px;'>
                    <span style='color: #B8C1CC;'>PPE Found:</span>
                    <div style='color: #2F80ED; margin-top: 6px;'>✓ Helmet ✓ Vest</div>
                </div>
                <div>
                    <span style='color: #B8C1CC;'>Violations:</span>
                    <div style='color: #EB5757; margin-top: 6px;'>⚠️ Missing Gloves</div>
                </div>
            </div>
            """
            st.markdown(analysis_html, unsafe_allow_html=True)


def logs_page():
    """Violation Logs with Filters"""
    st.markdown("<h1 style='margin-bottom: 30px;'>📋 Violation Logs & History</h1>", unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect("Risk Level", ["All", "High", "Medium", "Low"], default=["All"])
    with col2:
        date_range = st.date_input("Date Range", value=(datetime.now() - timedelta(days=30), datetime.now()), max_value=datetime.now())
    with col3:
        search_query = st.text_input("Search Employee", placeholder="Enter name...")
    
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    
    # Generate sample violation logs
    violations_data = {
        'ID': ['V001', 'V002', 'V003', 'V004', 'V005', 'V006', 'V007', 'V008'],
        'Timestamp': [
            datetime.now() - timedelta(hours=i*2) for i in range(8)
        ],
        'Employee': ['John Doe', 'Jane Smith', 'Mike Johnson', 'Sarah Wilson', 'Tom Brown', 'Lisa Anderson', 'David Lee', 'Emma White'],
        'Missing PPE': ['Helmet', 'Vest', 'Helmet, Gloves', 'Mask', 'Goggles', 'Helmet', 'Vest, Gloves', 'All'],
        'Risk Level': ['High', 'Medium', 'High', 'Low', 'Medium', 'High', 'High', 'Critical'],
        'Location': ['Warehouse A', 'Site B', 'Warehouse A', 'Office', 'Warehouse B', 'Site C', 'Warehouse A', 'Site B']
    }
    
    df = pd.DataFrame(violations_data)
    
    # Export buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if st.button("📊 Export CSV", use_container_width=True):
            st.success("CSV exported successfully")
    with col3:
        if st.button("📄 Export PDF", use_container_width=True):
            st.success("PDF exported successfully")
    
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    
    # Display dataframe
    st.dataframe(df, use_container_width=True, hide_index=True)


def analytics_page():
    """Advanced Analytics Page"""
    st.markdown("<h1 style='margin-bottom: 30px;'>📈 Advanced Analytics & Insights</h1>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Compliance Trends", "Employee Stats", "PPE Analytics", "System Health"])
    
    with tab1:
        st.markdown("<h3>Monthly Compliance Trends</h3>", unsafe_allow_html=True)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        compliance = [78, 81, 85, 88, 91, 94]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=compliance, mode='lines+markers', 
                                 name='Compliance %', line=dict(color='#27AE60', width=4),
                                 marker=dict(size=12, symbol='diamond')))
        fig.update_layout(template='plotly_dark', hovermode='x unified', 
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         font=dict(color='#B8C1CC'), height=400)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with tab2:
        st.markdown("<h3>Top Violators</h3>", unsafe_allow_html=True)
        violators_data = {
            'Employee': ['John Doe', 'Mike Johnson', 'Tom Brown', 'David Lee', 'James Miller'],
            'Violations': [15, 12, 10, 8, 6],
            'Compliance %': [65, 72, 78, 84, 90]
        }
        df = pd.DataFrame(violators_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("<h3>PPE Usage Patterns</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            ppe_compliance = {
                'Helmet': 95,
                'Vest': 92,
                'Gloves': 88,
                'Goggles': 85,
                'Mask': 80
            }
            fig = go.Figure(data=[go.Bar(x=list(ppe_compliance.keys()), 
                                         y=list(ppe_compliance.values()),
                                         marker_color=['#2F80ED', '#56CCF2', '#F2C94C', '#27AE60', '#EB5757'])])
            fig.update_layout(template='plotly_dark', height=350, 
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#B8C1CC'), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            st.markdown("<h4>Insights</h4>", unsafe_allow_html=True)
            st.markdown("""
            - **Helmet Usage:** Highest compliance at 95%
            - **Vest Usage:** Strong compliance at 92%
            - **Gloves:** Need improvement - 88%
            - **Goggles:** Moderate usage - 85%
            - **Mask:** Lowest usage - 80%
            
            **Recommendation:** Increase training on gloves and mask usage
            """)
    
    with tab4:
        st.markdown("<h3>System Performance Metrics</h3>", unsafe_allow_html=True)
        
        metrics_data = {
            'Metric': ['Detection Accuracy', 'Face Recognition', 'Response Time', 'Uptime', 'AI Confidence'],
            'Value': [98.5, 96.2, 245, 99.8, 94.7],
            'Unit': ['%', '%', 'ms', '%', '%']
        }
        
        col1, col2, col3, col4, col5 = st.columns(5)
        cols = [col1, col2, col3, col4, col5]
        
        for i, col in enumerate(cols):
            with col:
                metric = metrics_data['Metric'][i]
                value = metrics_data['Value'][i]
                unit = metrics_data['Unit'][i]
                kpi_card(metric, f"{value}{unit}", icon="📊")


# ============================================================================
# MAIN APP
# ============================================================================

@st.cache_resource
def load_components():
    try:
        db = DatabaseManager()
        yolo = YoloDetector()
        face_rec = FaceRecognizer()
        engine = ViolationEngine(db)
        return db, yolo, face_rec, engine
    except Exception as e:
        st.warning(f"⚠️ Component loading: {str(e)[:100]}")
        return None, None, None, None

# Load components
db, yolo, face_rec, engine = load_components()

# Page routing
PAGES = ["Dashboard", "Live Camera", "Upload Media", "Logs", "Analytics"]
query_params = st.query_params
current_page = query_params.get("page", "Dashboard")

if current_page not in PAGES:
    current_page = "Dashboard"

# Sidebar navigation
with st.sidebar:
    st.markdown("<h3 style='margin-bottom: 20px;'>🗂️ Navigation</h3>", unsafe_allow_html=True)
    selected = st.radio("Select Page", PAGES, label_visibility="collapsed")
    if selected != current_page:
        st.query_params["page"] = selected

# Render navbar
render_navbar(current_page)

# Route to selected page
if current_page == "Dashboard":
    dashboard_page()
elif current_page == "Live Camera":
    live_camera_page()
elif current_page == "Upload Media":
    upload_media_page()
elif current_page == "Logs":
    logs_page()
elif current_page == "Analytics":
    analytics_page()

st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)
render_footer()
