#!/usr/bin/env python3
"""
YOLO Classifier for tape color detection (inference only).
This is a minimal file for running the app - no training code.
"""

import os
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Error: ultralytics not installed. Run: pip install ultralytics")


class TapeClassifier:
    """YOLO-based classifier for tape color detection"""
    
    def __init__(self, model_path: str = "tape_classifier.pt"):
        self.model_path = model_path
        self.categories = ["blank", "no_package", "red", "yellow"]
        self.model = None
        
    def load(self) -> bool:
        """Load trained YOLO model"""
        if not YOLO_AVAILABLE:
            print("Error: ultralytics not installed")
            return False
            
        if not os.path.exists(self.model_path):
            print(f"Error: Model not found at {self.model_path}")
            return False
        
        try:
            self.model = YOLO(self.model_path)
            if hasattr(self.model, 'names'):
                self.categories = list(self.model.names.values())
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, img: np.ndarray) -> tuple:
        """
        Predict category for an image.
        Returns: (category_name, confidence)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        results = self.model(img, verbose=False)
        
        if len(results) > 0 and results[0].probs is not None:
            probs = results[0].probs
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            category = self.categories[top1_idx]
            return category, top1_conf
        
        return "no_package", 0.0
