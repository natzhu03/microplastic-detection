#not detecting

import random
import cv2
from ultralytics import YOLO
from tracker import Tracker
import pyrealsense2 as rs
import numpy as np
from yolov8 import YOLOv8
import argparse
import os.path

parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
# Parse the command line arguments to an object
args = parser.parse_args()
# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()

try:

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, args.input)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

    pipeline.start(config)

    # model_path = "models/yolov8m.onnx"
    model_path = "models/best.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    colorizer = rs.colorizer()

    cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)

    tracker = Tracker()
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]
    detection_threshold = 0.5

    font = cv2.FONT_HERSHEY_SIMPLEX 

    try:
        while True:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            depth_frame = frames.get_depth_frame()
            depth_color_frame = colorizer.colorize(depth_frame)
            depth_image = np.asanyarray(depth_color_frame.get_data())

            frame = np.asanyarray(color_frame.get_data())
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

                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                depth_value = depth_frame.get_distance(x_center, y_center)

                class_id = int(class_id)
                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score])
                
                tracker.update(frame, np.asarray(detections))
                #this should achieve the same effect?
                depth_values = []
                for track in tracker.tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = bbox
                    track_id = track.track_id
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

                    depth_value = depth_frame.get_distance(int((x1 + x2) / 2), int((y1 + y2) / 2))
                    depth_values.append(depth_value)

                    # text = f"ID: {track_id}, Depth: {depth_values[track_id-1]:.2f}m"
                    # cv2.putText(frame, text, (int(x1), int(y1) - 10), font, 0.9, (255, 255, 255), 2)
                    text = f"ID: {track_id}"
                    cv2.putText(frame, text, (int(x1), int(y1) - 10), font, 0.9, (255, 255, 255), 2)
                    #not sure why the ID is so high?

                    print("Center:", x_center, ",", y_center, "Distance:", depth_value, "ID:", track_id)


            cv2.imshow("Tracking", frame)

            
            combined_img = yolov8_detector.draw_detections(color_image)
            cv2.imshow("Detected Objects", combined_img)
            cv2.imshow("Depth Stream", depth_image)
            cv2.imshow("Color Stream", color_image)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

finally:
    pass