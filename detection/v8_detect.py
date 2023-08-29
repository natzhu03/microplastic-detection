# import os
# import cv2
# from yolov8 import YOLOv8
# import argparse

# def process_video(video_path, output_folder):
#     cap = cv2.VideoCapture(video_path)
#     start_time = 5  # skip first {start_time} seconds
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

#     # Initialize YOLOv7 model
#     model_path = "models/best2d.onnx"
#     yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

#     # Get video file name without extension
#     video_name = os.path.splitext(os.path.basename(video_path))[0]

#     # Create the output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     # Output video path
#     output_video_path = os.path.join(output_folder, f"{video_name}_output.avi")
#     out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS),
#                           (3840, 2160))

#     cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
#     while cap.isOpened():

#         # Press key q to stop
#         if cv2.waitKey(1) == ord('q'):
#             break

#         try:
#             # Read frame from the video
#             ret, frame = cap.read()
#             if not ret:
#                 break
#         except Exception as e:
#             print(e)
#             continue

#         # Update object localizer
#         boxes, scores, class_ids = yolov8_detector(frame)

#         combined_img = yolov8_detector.draw_detections(frame)
#         cv2.imshow("Detected Objects", combined_img)
#         out.write(combined_img)

#     out.release()
#     cap.release()

# def main():
#     parser = argparse.ArgumentParser(description='Run YOLOv8 on videos in a folder')
#     parser.add_argument('video_folder', type=str, help='Path to the folder containing videos')
#     args = parser.parse_args()

#     video_folder = args.video_folder
#     if not os.path.isdir(video_folder):
#         print(f"Error: The specified path '{video_folder}' is not a directory.")
#         return

#     video_files = [file for file in os.listdir(video_folder) if file.endswith('.mp4')]
#     if not video_files:
#         print(f"No video files found in '{video_folder}'.")
#         return

#     output_folder = "runs_detect"  # Replace with the path where you want to save the output videos
#     for video_file in video_files:
#         video_path = os.path.join(video_folder, video_file)
#         process_video(video_path, output_folder)

# if __name__ == "__main__":
#     main()

import cv2
from cap_from_youtube import cap_from_youtube

from yolov8 import YOLOv8

# # Initialize video
# cap = cv2.VideoCapture("input.mp4")

# videoUrl = 'https://youtu.be/Snyg0RqpVxY'
#cap = cap_from_youtube(videoUrl, resolution='720p')

cap = cv2.VideoCapture("05_06_23_13_50_05_958647_10cs_5mm_455.mp4")
start_time = 5 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (3840, 2160))

# Initialize YOLOv7 model
model_path = "models/best2d.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.1, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)
    # out.write(combined_img)

# out.release()