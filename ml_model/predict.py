
from PIL import Image
from ultralytics import YOLO

model = YOLO('runs/detect/train5/weights/best.pt')
results = model.predict('datasets/extended_dt/test_tr/images/train/sep_bl_sh(180,68).jpg')  # results list
for r in results:
    im_array = r.plot()  
    im = Image.fromarray(im_array[..., ::-1]) 
    im.show() 
    boxes = r.boxes.cpu().numpy()
    print(boxes.xyxy)

