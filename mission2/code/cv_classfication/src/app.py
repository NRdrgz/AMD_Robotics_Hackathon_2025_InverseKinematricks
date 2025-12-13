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

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class ConveyorDetector:
    
    def __init__(self, camera_id: int = 0, model_path: str = None):
        # Default model path relative to script directory
        if model_path is None:
            model_path = os.path.join(SCRIPT_DIR, "tape_classifier.pt")
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
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return False
        
        frame_diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        self.prev_gray = cv2.addWeighted(self.prev_gray, 0.7, gray, 0.3, 0)
        
        return cv2.countNonZero(thresh) > self.motion_area_threshold
    
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
<html>
<head>
    <title>Conveyor Package Detector</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: white; }
        h1 { color: #00d4ff; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .video-box { flex: 2; min-width: 500px; }
        .video-box img { width: 100%; border-radius: 8px; border: 2px solid #333; }
        .status-box { flex: 1; min-width: 250px; background: #16213e; padding: 20px; border-radius: 8px; }
        .status-item { margin: 15px 0; padding: 15px; border-radius: 8px; background: #0f3460; }
        .status-item.active { background: #00d44f; color: black; }
        .status-item.delivering { background: #ff9800; color: black; }
        .status-item.yellow { background: #ffd700; color: black; }
        .status-item.red { background: #ff4444; color: white; }
        .btn { padding: 12px 24px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .btn-start { background: #00d44f; color: black; }
        .btn-stop { background: #ff4444; color: white; }
        .btn:hover { opacity: 0.8; }
        .label { font-size: 12px; color: #888; text-transform: uppercase; }
        .value { font-size: 24px; font-weight: bold; margin-top: 5px; }
    </style>
</head>
<body>
    <h1>📦 Conveyor Package Detector</h1>
    <div style="margin-bottom: 20px;">
        <button class="btn btn-start" onclick="startDetector()">▶ Start</button>
        <button class="btn btn-stop" onclick="stopDetector()">⏹ Stop</button>
    </div>
    <div class="container">
        <div class="video-box">
            <img id="video-stream" src="/video_feed" alt="Video Stream">
        </div>
        <div class="status-box">
            <h2>Status</h2>
            <div id="package-status" class="status-item">
                <div class="label">Package</div>
                <div class="value" id="package-value">No Package</div>
            </div>
            <div id="color-status" class="status-item">
                <div class="label">Sorting Color</div>
                <div class="value" id="color-value">-</div>
            </div>
            <div class="status-item">
                <div class="label">System</div>
                <div class="value" id="system-value">Stopped</div>
            </div>
        </div>
    </div>
    <script>
        function updateStatus() {
            fetch('/api/status').then(r => r.json()).then(data => {
                const pkgEl = document.getElementById('package-status');
                const pkgVal = document.getElementById('package-value');
                if (data.is_delivering) {
                    pkgEl.className = 'status-item delivering';
                    pkgVal.textContent = '🚚 DELIVERING';
                } else if (data.package_detected) {
                    pkgEl.className = 'status-item active';
                    pkgVal.textContent = '📦 DETECTED';
                } else {
                    pkgEl.className = 'status-item';
                    pkgVal.textContent = 'No Package';
                }
                
                const colorEl = document.getElementById('color-status');
                const colorVal = document.getElementById('color-value');
                const confStr = data.confidence > 0 ? ` (${data.confidence}%)` : '';
                if (data.color === 'YELLOW') {
                    colorEl.className = 'status-item yellow';
                    colorVal.textContent = '🟡 YELLOW' + confStr;
                } else if (data.color === 'RED') {
                    colorEl.className = 'status-item red';
                    colorVal.textContent = '🔴 RED' + confStr;
                } else if (data.color === 'BLANK') {
                    colorEl.className = 'status-item active';
                    colorVal.textContent = '⬜ BLANK' + confStr;
                } else {
                    colorEl.className = 'status-item';
                    colorVal.textContent = '-';
                }
                
                document.getElementById('system-value').textContent = data.running ? '🟢 Running' : '⚫ Stopped';
            });
        }
        function startDetector() {
            fetch('/api/start', {method: 'POST'}).then(() => {
                document.getElementById('video-stream').src = '/video_feed?' + Date.now();
            });
        }
        function stopDetector() { fetch('/api/stop', {method: 'POST'}); }
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
