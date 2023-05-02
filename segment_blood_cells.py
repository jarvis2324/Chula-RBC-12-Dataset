from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sam_checkpoint = r"C:\Users\desir\Documents\computer vision\SegmentAnything\models\sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

img=cv2.imread(r"C:\Users\desir\Downloads\1 (1).jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
mask_predictor=SamPredictor(sam)
mask_predictor.set_image(img)
ann=pd.read_csv(r'C:\Users\desir\Documents\computer vision\SegmentAnything\annotations\1.txt',header=None,sep=" ")

coords= np.array(ann.iloc[:, 1:3])  # Use a numpy array instead of a list

labels = np.ones(len(ann))
    

masks, scores, logits = mask_predictor.predict(
    point_coords=np.array(coords),
    point_labels=np.array(labels),
    multimask_output=True
)
mask=np.uint8(masks[1])*255
plt.imshow(mask,cmap='gray')
