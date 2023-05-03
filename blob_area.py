import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the segmentation mask as a binary image
mask = cv2.imread('segmentation_mask.png', 0)
if mask.dtype==bool:
    print('e')
    mask=np.uint8(mask)*255

# Find the connected components (blobs) in the mask
_, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

filtered_labels = np.zeros_like(labels)
i=0
for label, stat in enumerate(stats):
    area = stat[4]
    if area >= 100:
        i+=1
        filtered_labels[labels == label] = label

# Generate a random color for each filtered blob label
colors = np.random.randint(0, 255, size=(np.max(filtered_labels)+1 , 3), dtype=np.uint8)

# Map each filtered label to a different color
colored_labels = colors[filtered_labels]
plt.imshow(colored_labels)
