import pyrealsense2 as rs
import numpy as np
import cv2
from yolov8 import YOLOv8


pipeline = rs.pipeline()
config = rs.config()

    #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    #config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

    # config.enable_record_to_file(save_path)
pipeline.start(config)

# Initialize YOLOv7 model
# model_path = "models/yolov8m.onnx"
model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()) 

        #cv2.imshow('Color Image', color_image)

         # Update object localizer
        boxes, scores, class_ids = yolov8_detector(color_image)
        for box in boxes:
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            print("Center:", x_center, y_center, "Depth:", depth_image[int(y_center), int(x_center)])

        combined_img = yolov8_detector.draw_detections(color_image)
        cv2.imshow("Detected Objects", combined_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()