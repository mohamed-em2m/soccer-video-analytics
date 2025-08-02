from typing import List
import numpy as np
import pandas as pd
import torch
from inference.base_detector import BaseDetector
from ultralytics import YOLO
    
class YoloV5(BaseDetector):
    def __init__(self, model_path: str = None, device: str = None):
        # 1. pick your device
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        if model_path:
            try:
                
                print(f"Loading model {model_path}")
                self.model = YOLO(model_path)
            
            except:
                
                print(f"Failed load model {model_path} load yolov5x insteaded")
                self.model = YOLO("yolov5x.pt")

        else:
            
            print("Loading yolov5x model")
            self.model = YOLO("yolov5x.pt")
            
    def predict(self, input_image: List[np.ndarray]) -> pd.DataFrame:
        """
        Predicts the bounding boxes of the objects in the image
        Parameters
        ----------
        input_image : List[np.ndarray]
            List of input images
        Returns
        -------
        pd.DataFrame
            DataFrame containing the bounding boxes
        """
        results = self.model(input_image, imgsz=640)
        
        # Convert ultralytics results to pandas DataFrame
        detections = []
        for i, result in enumerate(results):
            boxes = result.boxes
            if boxes is not None:
                for j in range(len(boxes)):
                    # Extract box coordinates and other info
                    x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy()
                    confidence = boxes.conf[j].cpu().numpy()
                    class_id = int(boxes.cls[j].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detections.append({
                        'xmin': x1,
                        'ymin': y1,
                        'xmax': x2,
                        'ymax': y2,
                        'confidence': confidence,
                        'class': class_id,
                        'name': class_name
                    })
        
        return pd.DataFrame(detections)
