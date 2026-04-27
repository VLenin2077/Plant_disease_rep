from ultralytics import YOLO

# Load a model

model = YOLO('yolov8s.pt') 

# Train the model
results = model.train(data='plant.yaml',
                      epochs=50,
                      imgsz=256,
                      batch=8,
                      optimizer='Adam',
                      lr0=1E-3
                      )
