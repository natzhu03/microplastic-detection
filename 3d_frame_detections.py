# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
from yolov8 import YOLOv8


# Create object for parsing command-line options
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
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)

    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

    # Start streaming from file
    pipeline.start(config)

    # Initialize YOLOv7 model
    # model_path = "models/yolov8m.onnx"
    model_path = "models/best.onnx"

    yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)
   
    # Create colorizer object
    colorizer = rs.colorizer()

    # Streaming loop
    try:
        while True:
            # Get frameset of depth
            frames = pipeline.wait_for_frames()

            # Get depth frame
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            color_image = np.asanyarray(color_frame.get_data())

            # Colorize depth frame to jet colormap
            depth_color_frame = colorizer.colorize(depth_frame)

            # Convert depth_frame to numpy array to render image in opencv
            depth_image = np.asanyarray(depth_color_frame.get_data())

            # depth_image = np.asanyarray(depth_frame.get_data()) 

            ###
            boxes, scores, class_ids = yolov8_detector(color_image)
      
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)

                #tuples
                # north = (x_center, int(y1-((y1-y2)/4)))
                # south = (x_center, int(y2+((y1-y2)/4)))
                # east = (int(x2-((x2-x1)/4)), y_center)
                # west = (int(x1+((x2-x1)/4)), y_center)
                # center = (x_center, y_center)

                # nsew = [north, south, east, west, center] #list of tuples

                # summ = 0
                # num = 0

                # for x,y in nsew:
                #     depth_value = depth_frame.get_distance(x,y)  
                #     if depth_value != 0:
                #         summ += depth_value
                #         num += 1

                #iterating through all the values
                summ = 0
                num = 0
                for y in range(y1, y2 + 1):
    # Iterate through the columns
                    for x in range(x1, x2 + 1):
                        depth_value = depth_frame.get_distance(x,y)  
                        if depth_value != 0:
                            summ += depth_value
                            num += 1

                # depth_value = depth_frame.get_distance(x_center, y_center)  
                if num != 0:
                    depth_value = summ/num  
                else:
                    depth_value = 0
                    #possible do an average of different values, and eliminate values of 0?
                    #but even with the bounding box, only want value of 0

                    #could get value at 

                print("Center:", x_center, ",", y_center, "Distance:", depth_value)

                 # Render image in opencv window
                color_image = cv2.circle(color_image, (x_center,y_center), radius=8, color=(255, 0, 0), thickness=2)
                # depth_image = cv2.circle(depth_image, (x_center,y_center), radius=8, color=(255, 0, 0), thickness=2)

                

            combined_img = yolov8_detector.draw_detections(color_image)
            cv2.imshow("Detected Objects", combined_img)

           
        
            cv2.imshow("Depth Stream", depth_image)
            cv2.imshow("Color Stream", color_image)

            # key = cv2.waitKey(1)
            # if pressed escape exit program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

finally:
    pass
