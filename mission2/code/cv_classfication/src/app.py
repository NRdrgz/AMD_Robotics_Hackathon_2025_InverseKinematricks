#!/usr/bin/env python3
"""
Web App for Conveyor Package Detection
Uses YOLO classifier for color detection on the full frame.
"""

import cv2
import numpy as np
import time
import threading
import os
from collections import deque
from flask import Flask, jsonify, Response, render_template_string
from typing import Optional
from enum import Enum

# Try to import from classifier.py first (inference-only), fallback to train_classifier.py
try:
    from classifier import TapeClassifier
except ImportError:
    from train_classifier import TapeClassifier


class TapeColor(Enum):
    YELLOW = "YELLOW"
    RED = "RED"
    BLANK = "BLANK"
    NO_PACKAGE = "NO_PACKAGE"


app = Flask(__name__)


class ConveyorDetector:
    
    def __init__(self, camera_id: int = 0, model_path: str = "tape_classifier.pt"):
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.thread = None
        
        # YOLO Classifier
        self.classifier = None
        self.model_path = model_path
        self._load_classifier()
        
        # Status tracking
        self.package_detected = False
        self.is_delivering = False
        self.color_history = deque(maxlen=15)
        self.confidence_history = deque(maxlen=15)
        self.current_color = TapeColor.NO_PACKAGE
        self.current_confidence = 0.0
        self.last_frame = None
        self.lock = threading.Lock()
        
        # Motion detection
        self.prev_gray = None
        self.motion_threshold = 25
        self.motion_area_threshold = 5000
    
    def _load_classifier(self):
        if os.path.exists(self.model_path):
            try:
                self.classifier = TapeClassifier(model_path=self.model_path)
                if self.classifier.load():
                    print(f"✅ YOLO classifier loaded from {self.model_path}")
                else:
                    print(f"⚠️  Failed to load YOLO classifier")
            except Exception as e:
                print(f"⚠️  Error loading classifier: {e}")
        else:
            print(f"⚠️  No model found at {self.model_path}")
        
    def start(self) -> bool:
        if self.running:
            return True
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        return True
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def _run_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            color, confidence = self._detect_color(frame)
            motion_detected = self._detect_motion(frame)
            
            with self.lock:
                self.color_history.append(color)
                self.confidence_history.append(confidence)
                self._update_status()
                self.is_delivering = motion_detected and self.package_detected
                self.last_frame = self._annotate_frame(frame)
            
            time.sleep(0.03)
    
    def _detect_motion(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Reset prev_gray if None or size mismatch
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return False
        
        try:
            frame_diff = cv2.absdiff(self.prev_gray, gray)
            _, thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
            self.prev_gray = cv2.addWeighted(self.prev_gray, 0.7, gray, 0.3, 0)
            return cv2.countNonZero(thresh) > self.motion_area_threshold
        except Exception:
            self.prev_gray = gray
            return False
    
    def _detect_color(self, frame: np.ndarray) -> tuple:
        if self.classifier is not None:
            try:
                category, confidence = self.classifier.predict(frame)
                color_map = {
                    "yellow": TapeColor.YELLOW, 
                    "red": TapeColor.RED, 
                    "blank": TapeColor.BLANK,
                    "no_package": TapeColor.NO_PACKAGE
                }
                return color_map.get(category, TapeColor.NO_PACKAGE), confidence
            except:
                pass
        return TapeColor.NO_PACKAGE, 0.0
    
    def _update_status(self):
        if len(self.color_history) < 3:
            self.current_color = TapeColor.NO_PACKAGE
            self.current_confidence = 0.0
            self.package_detected = False
            return
        
        total = len(self.color_history)
        yellow_count = sum(1 for c in self.color_history if c == TapeColor.YELLOW)
        red_count = sum(1 for c in self.color_history if c == TapeColor.RED)
        blank_count = sum(1 for c in self.color_history if c == TapeColor.BLANK)
        no_pkg_count = sum(1 for c in self.color_history if c == TapeColor.NO_PACKAGE)
        
        avg_confidence = sum(self.confidence_history) / len(self.confidence_history) if self.confidence_history else 0.0
        
        if yellow_count / total > 0.5:
            self.current_color = TapeColor.YELLOW
            self.current_confidence = avg_confidence
            self.package_detected = True
        elif red_count / total > 0.5:
            self.current_color = TapeColor.RED
            self.current_confidence = avg_confidence
            self.package_detected = True
        elif blank_count / total > 0.5:
            self.current_color = TapeColor.BLANK
            self.current_confidence = avg_confidence
            self.package_detected = True
        elif no_pkg_count / total > 0.5:
            self.current_color = TapeColor.NO_PACKAGE
            self.current_confidence = avg_confidence
            self.package_detected = False
    
    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        if self.package_detected:
            status = "DELIVERING" if self.is_delivering else "DETECTED"
            status_color = (0, 200, 255) if self.is_delivering else (0, 255, 0)
            conf_str = f" ({self.current_confidence:.0%})" if self.current_confidence > 0 else ""
            color_status = f"Color: {self.current_color.value}{conf_str}"
            
            border_colors = {
                TapeColor.YELLOW: (0, 255, 255),
                TapeColor.RED: (0, 0, 255),
                TapeColor.BLANK: (0, 255, 0)
            }
            border_color = border_colors.get(self.current_color, (0, 255, 0))
            cv2.rectangle(annotated, (5, 5), (w-5, h-5), border_color, 8)
        else:
            status = "No Package"
            status_color = (150, 150, 150)
            color_status = ""
        
        cv2.rectangle(annotated, (10, 10), (380, 80), (0, 0, 0), -1)
        cv2.putText(annotated, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
        if color_status:
            text_colors = {"YELLOW": (0, 255, 255), "RED": (0, 0, 255), "BLANK": (200, 200, 200)}
            text_color = text_colors.get(self.current_color.value, (255, 255, 255))
            cv2.putText(annotated, color_status, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        return annotated
    
    def get_status(self) -> dict:
        with self.lock:
            color_value = None
            if self.current_color in [TapeColor.YELLOW, TapeColor.RED, TapeColor.BLANK]:
                color_value = self.current_color.value
            
            status = "DELIVERING" if self.is_delivering else ("DETECTED" if self.package_detected else "NO_PACKAGE")
            
            return {
                "status": status,
                "package_detected": self.package_detected,
                "is_delivering": self.is_delivering,
                "color": color_value,
                "confidence": round(self.current_confidence * 100, 1),
                "running": self.running
            }
    
    def get_frame_jpeg(self) -> Optional[bytes]:
        with self.lock:
            if self.last_frame is None:
                return None
            _, jpeg = cv2.imencode('.jpg', self.last_frame)
            return jpeg.tobytes()


detector = ConveyorDetector(camera_id=0)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flip and Ship</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        
        .header {
            background: rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding: 16px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #00d4ff, #0099ff);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        
        .logo-text {
            font-size: 20px;
            font-weight: 700;
            background: linear-gradient(135deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .connection-status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            font-size: 13px;
        }
        
        .connection-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #666;
            transition: all 0.3s;
        }
        
        .connection-dot.online {
            background: #00ff88;
            box-shadow: 0 0 10px #00ff88;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .main-content {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }
        
        .controls {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
        }
        
        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-size: 15px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
            font-family: inherit;
        }
        
        .btn-start {
            background: linear-gradient(135deg, #00d44f, #00aa3f);
            color: white;
            box-shadow: 0 4px 15px rgba(0, 212, 79, 0.3);
        }
        
        .btn-start:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 212, 79, 0.4);
        }
        
        .btn-stop {
            background: linear-gradient(135deg, #ff4757, #ff3344);
            color: white;
            box-shadow: 0 4px 15px rgba(255, 71, 87, 0.3);
        }
        
        .btn-stop:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 71, 87, 0.4);
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 24px;
        }
        
        @media (max-width: 900px) {
            .dashboard { grid-template-columns: 1fr; }
        }
        
        .video-container {
            background: rgba(0,0,0,0.3);
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .video-header {
            padding: 12px 16px;
            background: rgba(0,0,0,0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        
        .video-title {
            font-size: 14px;
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .live-badge {
            background: #ff4757;
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            animation: pulse 1.5s infinite;
        }
        
        .video-frame {
            width: 100%;
            display: block;
            aspect-ratio: 16/9;
            object-fit: cover;
            background: #000;
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        
        .status-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 20px;
            transition: all 0.3s;
        }
        
        .status-card.highlight {
            border-color: rgba(0, 212, 79, 0.5);
            box-shadow: 0 0 30px rgba(0, 212, 79, 0.1);
        }
        
        .status-card.delivering {
            border-color: rgba(255, 152, 0, 0.5);
            box-shadow: 0 0 30px rgba(255, 152, 0, 0.1);
            animation: deliverPulse 1s infinite;
        }
        
        @keyframes deliverPulse {
            0%, 100% { box-shadow: 0 0 30px rgba(255, 152, 0, 0.1); }
            50% { box-shadow: 0 0 40px rgba(255, 152, 0, 0.25); }
        }
        
        .card-label {
            font-size: 11px;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }
        
        .package-status {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .status-icon {
            width: 56px;
            height: 56px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            background: rgba(255,255,255,0.05);
            transition: all 0.3s;
        }
        
        .status-icon.detected {
            background: linear-gradient(135deg, #00d44f, #00aa3f);
            box-shadow: 0 4px 20px rgba(0, 212, 79, 0.4);
        }
        
        .status-icon.delivering {
            background: linear-gradient(135deg, #ff9800, #ff6d00);
            box-shadow: 0 4px 20px rgba(255, 152, 0, 0.4);
        }
        
        .status-text h3 {
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        
        .status-text p {
            font-size: 13px;
            color: #666;
        }
        
        .color-display {
            text-align: center;
            padding: 24px;
        }
        
        .color-badge {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 16px 32px;
            border-radius: 50px;
            font-size: 20px;
            font-weight: 700;
            background: rgba(255,255,255,0.05);
            transition: all 0.3s;
        }
        
        .color-badge.yellow {
            background: linear-gradient(135deg, #ffd700, #ffaa00);
            color: #1a1a1a;
            box-shadow: 0 4px 25px rgba(255, 215, 0, 0.4);
        }
        
        .color-badge.red {
            background: linear-gradient(135deg, #ff4757, #ff2233);
            color: white;
            box-shadow: 0 4px 25px rgba(255, 71, 87, 0.4);
        }
        
        .color-badge.blank {
            background: linear-gradient(135deg, #a0a0a0, #888);
            color: white;
            box-shadow: 0 4px 25px rgba(160, 160, 160, 0.3);
        }
        
        .confidence-bar {
            margin-top: 16px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            border-radius: 8px;
            transition: width 0.3s;
        }
        
        .confidence-text {
            margin-top: 8px;
            font-size: 13px;
            color: #666;
        }
        
        .system-status {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .system-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .system-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #666;
        }
        
        .system-dot.running {
            background: #00ff88;
            box-shadow: 0 0 15px #00ff88;
        }
        
        .system-label {
            font-size: 18px;
            font-weight: 600;
        }
        
        
        .footer {
            margin-top: 24px;
            padding: 16px;
            text-align: center;
            color: #444;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            <div class="logo-icon">📦</div>
            <span class="logo-text">Flip and Ship</span>
        </div>
        <div class="connection-status">
            <div class="connection-dot" id="connection-dot"></div>
            <span id="connection-text">Connecting...</span>
        </div>
    </div>
    
    <div class="main-content">
        <div class="controls">
            <button class="btn btn-start" onclick="startDetector()">
                <span>▶</span> Start Detection
            </button>
            <button class="btn btn-stop" onclick="stopDetector()">
                <span>⏹</span> Stop
            </button>
        </div>
        
        <div class="dashboard">
            <div class="video-container">
                <div class="video-header">
                    <span class="video-title">Live Camera Feed</span>
                    <span class="live-badge" id="live-badge" style="display:none;">● LIVE</span>
                </div>
                <img id="video-stream" class="video-frame" src="/video_feed" alt="Video Stream">
            </div>
            
            <div class="sidebar">
                <div class="status-card" id="package-card">
                    <div class="card-label">Package Status</div>
                    <div class="package-status">
                        <div class="status-icon" id="status-icon">📭</div>
                        <div class="status-text">
                            <h3 id="package-value">No Package</h3>
                            <p id="package-subtext">Waiting for package...</p>
                        </div>
                    </div>
                </div>
                
                <div class="status-card" id="color-card">
                    <div class="card-label">Sorting</div>
                    <div class="color-display">
                        <div class="color-badge" id="color-badge">
                            <span>—</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" id="confidence-fill" style="width: 0%"></div>
                        </div>
                        <div class="confidence-text" id="confidence-text">Confidence: —</div>
                    </div>
                </div>
                
                <div class="status-card">
                    <div class="card-label">System</div>
                    <div class="system-status">
                        <div class="system-indicator">
                            <div class="system-dot" id="system-dot"></div>
                            <span class="system-label" id="system-value">Stopped</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            Powered by WOWROBOT
        </div>
    </div>
    
    <script>
        let isRunning = false;
        
        function updateStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    isRunning = data.running;
                    
                    // Connection status
                    const connDot = document.getElementById('connection-dot');
                    const connText = document.getElementById('connection-text');
                    connDot.className = 'connection-dot' + (data.running ? ' online' : '');
                    connText.textContent = data.running ? 'Connected' : 'Offline';
                    
                    // Live badge
                    document.getElementById('live-badge').style.display = data.running ? 'block' : 'none';
                    
                    // Package status
                    const pkgCard = document.getElementById('package-card');
                    const pkgIcon = document.getElementById('status-icon');
                    const pkgVal = document.getElementById('package-value');
                    const pkgSub = document.getElementById('package-subtext');
                    
                    if (data.is_delivering) {
                        pkgCard.className = 'status-card delivering';
                        pkgIcon.className = 'status-icon delivering';
                        pkgIcon.textContent = '🚚';
                        pkgVal.textContent = 'DELIVERING';
                        pkgSub.textContent = 'Package in motion';
                    } else if (data.package_detected) {
                        pkgCard.className = 'status-card highlight';
                        pkgIcon.className = 'status-icon detected';
                        pkgIcon.textContent = '📦';
                        pkgVal.textContent = 'DETECTED';
                        pkgSub.textContent = 'Package ready for sorting';
                    } else {
                        pkgCard.className = 'status-card';
                        pkgIcon.className = 'status-icon';
                        pkgIcon.textContent = '📭';
                        pkgVal.textContent = 'No Package';
                        pkgSub.textContent = 'Waiting for package...';
                    }
                    
                    // Color status
                    const colorBadge = document.getElementById('color-badge');
                    const confFill = document.getElementById('confidence-fill');
                    const confText = document.getElementById('confidence-text');
                    
                    if (data.color === 'YELLOW') {
                        colorBadge.className = 'color-badge yellow';
                        colorBadge.innerHTML = '🟡 YELLOW';
                    } else if (data.color === 'RED') {
                        colorBadge.className = 'color-badge red';
                        colorBadge.innerHTML = '🔴 RED';
                    } else if (data.color === 'BLANK') {
                        colorBadge.className = 'color-badge blank';
                        colorBadge.innerHTML = '⬜ NO CATEGORY';
                    } else {
                        colorBadge.className = 'color-badge';
                        colorBadge.innerHTML = '<span>—</span>';
                    }
                    
                    const conf = data.confidence || 0;
                    if (conf >= 50) {
                        confFill.style.width = conf + '%';
                        confText.textContent = `Confidence: ${conf}%`;
                        document.querySelector('.confidence-bar').style.display = 'block';
                    } else {
                        confFill.style.width = '0%';
                        confText.textContent = '';
                        document.querySelector('.confidence-bar').style.display = 'none';
                    }
                    
                    // System status
                    document.getElementById('system-dot').className = 'system-dot' + (data.running ? ' running' : '');
                    document.getElementById('system-value').textContent = data.running ? 'Running' : 'Stopped';
                })
                .catch(() => {
                    document.getElementById('connection-dot').className = 'connection-dot';
                    document.getElementById('connection-text').textContent = 'Disconnected';
                });
        }
        
        function startDetector() {
            fetch('/api/start', {method: 'POST'}).then(() => {
                document.getElementById('video-stream').src = '/video_feed?' + Date.now();
            });
        }
        
        function stopDetector() {
            fetch('/api/stop', {method: 'POST'});
        }
        
        setInterval(updateStatus, 200);
        updateStatus();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    return jsonify(detector.get_status())

@app.route('/api/start', methods=['POST'])
def api_start():
    return jsonify({"success": detector.start()})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    detector.stop()
    return jsonify({"success": True})

def generate_frames():
    while True:
        frame_bytes = detector.get_frame_jpeg()
        if frame_bytes:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  CONVEYOR DETECTOR WEB APP")
    print("="*50)
    print("\n  Open http://localhost:5001 in your browser")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
