import cv2
import numpy as np
from ultralytics import YOLO

# Load models
pt_model = YOLO('/Users/yash/Library/artikate/Artikate_Assignment/scripts/runs/detect/train-8/weights/best.pt')
onnx_model = YOLO('/Users/yash/Library/artikate/Artikate_Assignment/scripts/runs/detect/train-8/weights/best.onnx', task='detect')

img_path = '/Users/yash/Library/artikate/Artikate_Assignment/scripts/images/val/IMG20260906153940.jpg'

# Run inference
res_pt = pt_model(img_path)[0]
res_onnx = onnx_model(img_path)[0]

# Plot results
img_pt = res_pt.plot()
img_onnx = res_onnx.plot()

# Save to artifacts dir
cv2.imwrite('/Users/yash/.gemini/antigravity-ide/brain/95e32b1a-c21b-48de-9f6c-2300d9b5aaff/pytorch_result.jpg', img_pt)
cv2.imwrite('/Users/yash/.gemini/antigravity-ide/brain/95e32b1a-c21b-48de-9f6c-2300d9b5aaff/onnx_result.jpg', img_onnx)
print("Images saved.")
