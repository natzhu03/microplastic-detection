if __name__ == '__main__':
    
    from ultralytics import YOLO

    # Create a new YOLO model from scratch
    model = YOLO('yolov8n.yaml')

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='plastic.yaml', epochs=30, device=0)
    #for gpu, add "device=0" argument... currently not working

    # Evaluate the model's performance on the validation set
 
    # Perform object detection on an image using the model
    #results = model('https://ultralytics.com/images/bus.jpg')

    # Export the model to ONNX format
    success = model.export(format='onnx')