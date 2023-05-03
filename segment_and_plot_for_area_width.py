import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


class BlobAnalyzer:
    def __init__(self, sam_checkpoint, model_type, device):
        #Initialising model with Pre-trained weights
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.mask_predictor = None
        self.img = None
        self.mask = None
        self.areas = []
        self.widths = []

    def set_image(self, image_path):
        #Taking image
        self.img = cv2.imread(image_path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def set_annotations(self, annotations_path):
        #Reading annotations and considering labels as 1 Because we need segmentation for foreground 
        ann = pd.read_csv(annotations_path, header=None, sep=" ")
        self.coords = np.array(ann.iloc[:, 1:3])
        self.labels = np.ones(len(ann))

    def generate_mask(self):
        #loading image to model
        self.mask_predictor = SamPredictor(self.sam)
        self.mask_predictor.set_image(self.img)
        #Getting segmented mask
        masks, _, _ = self.mask_predictor.predict(
            point_coords=np.array(self.coords),
            point_labels=np.array(self.labels),
            multimask_output=True
        )
        #mask contains 3 output segments, but we will consider 2nd mask for our case
        self.mask = masks[1]
        if self.mask.dtype == bool:
            self.mask = np.uint8(self.mask) * 255

    def analyze_blobs(self):
        #Using contours for area and width calculations
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            width = rect[1][0]
            area = cv2.contourArea(contour)

            if area >= 50:
                self.areas.append(area)
                self.widths.append(width)

        area_hist, area_bins = np.histogram(self.areas, bins=50)
        width_hist, width_bins = np.histogram(self.widths, bins=50)

        area_centers = (area_bins[:-1] + area_bins[1:]) / 2
        width_centers = (width_bins[:-1] + width_bins[1:]) / 2
        #Ploting curves for area and width of Blood cell in a given input image
        plt.figure()
        plt.plot(area_centers, area_hist, color='blue', label='Area')
        plt.title("Distribution of Blob Areas")
        plt.xlabel("Area")
        plt.ylabel("Count")

        plt.figure()
        plt.plot(width_centers, width_hist, color='green', label='Width')
        plt.title("Distribution of Blob Widths")
        plt.xlabel("Width")
        plt.ylabel("Count")

        plt.figure()
        plt.imshow(self.img)
        plt.title("Original image")
        plt.show()


if __name__ == '__main__':
    sam_checkpoint = r"C:\Users\desir\Documents\computer vision\SegmentAnything\models\sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cpu"

    blob_analyzer = BlobAnalyzer(sam_checkpoint, model_type, device)
    blob_analyzer.set_image(r"C:\Users\desir\Downloads\1 (1).jpg")
    blob_analyzer.set_annotations(r'C:\Users\desir\Documents\computer vision\SegmentAnything\annotations\1.txt')
    blob_analyzer.generate_mask()
    blob_analyzer.analyze_blobs()
