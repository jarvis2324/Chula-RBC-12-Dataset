import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the segmentation mask as a binary image
mask = cv2.imread("path",0)
if mask.dtype==bool:
    mask=np.uint8(mask)*255

# Initialize empty arrays to store the areas and widths of the blobs
areas = []
widths = []
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Iterate through the filtered labels and calculate the area and width of each blob
for contour in contours:
    # Find the contour of the blob
    
    # Find the minimum area rectangle that bounds the contour
    rect = cv2.minAreaRect(contour)
    width = rect[1][0]
    area = cv2.contourArea(contour)
    
    # Append the area and width of the current blob to the arrays
    if area>=50:
        areas.append(area)
        widths.append(width)
area_hist, area_bins = np.histogram(areas, bins=50)
width_hist, width_bins = np.histogram(widths, bins=50)

# Compute the bin centers
area_centers = (area_bins[:-1] + area_bins[1:]) / 2
width_centers = (width_bins[:-1] + width_bins[1:]) / 2
# Plot the distribution of areas
plt.figure()
plt.plot(area_centers, area_hist, color='blue', label='Area')
plt.title("Distribution of Blob Areas")
plt.xlabel("Area")
plt.ylabel("Count")


# Plot the distribution of widths
plt.figure()
plt.plot(width_centers, width_hist, color='green', label='Width')
plt.title("Distribution of Blob Widths")
plt.xlabel("Width")
plt.ylabel("Count")



# Display the colored blob image
plt.figure()
plt.imshow(mask,cmap='gray')
plt.show()
