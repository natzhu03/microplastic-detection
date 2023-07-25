import random
import cv2
from ultralytics import YOLO
from tracker import Tracker
import pyrealsense2 as rs
import numpy as np
from yolov8 import YOLOv8

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

pipeline.start(config)

# model = YOLO("yolov8n.pt")
#change this to:
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)


tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]
detection_threshold = 0.5

font = cv2.FONT_HERSHEY_SIMPLEX 

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        # results = model(frame)
        boxes, scores, class_ids = yolov8_detector(frame)
        results = zip(boxes, scores, class_ids) #combine into touble

        # Iterate through the results
        for box, score, class_id in results:
            # Access individual box, score, and class_id
            detections = []
            x1 = int(box[0])
            x2 = int(box[2])
            y1 = int(box[1])
            y2 = int(box[3])
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])
            
            tracker.update(frame, np.asarray(detections))
            #this should achieve the same effect?
            
            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

                text = f"ID: {track_id}"
                cv2.putText(frame, text, (int(x1), int(y1) - 10), font, 0.9, (255, 255, 255), 2)
                #not sure why the ID is so high?


        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()