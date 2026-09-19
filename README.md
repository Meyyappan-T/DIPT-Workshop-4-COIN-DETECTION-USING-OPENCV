# Workshop-4 COIN DETECTION USING OPENCV

## AIM:
To detect and count the number of coins present in an image using image processing techniques such as grayscale conversion, thresholding, morphological operations, and blob detection with OpenCV.

## SOFTWARE REQUIREMENTS:
Python 3.x

OpenCV (cv2)

NumPy

Matplotlib

Jupyter Notebook or any Python IDE

## ALGORITHM:
 1. Read the Image: Load the coin image using cv2.imread().
 2. Convert to Grayscale: Convert the color image to grayscale for easier processing.
 3. Apply Thresholding: Convert the grayscale image into a binary image to separate coins from the background.
 4. Perform Morphological Operations: Use dilation and erosion to remove noise and enhance coin boundaries.
 5. Detect Blobs: Apply the SimpleBlobDetector to identify circular regions corresponding to coins and count them.

## Program:
### Name: MEYYAPPAN T
### Reg.No: 212223240086


```
import cv2
import matplotlib.pyplot as plt
import numpy as np

image=cv2.imread("Coins.png")
imageCopy = image.copy()

plt.imshow(image[:,:,::-1]);
plt.title("Original Image")
plt.show()
```
<img width="496" height="527" alt="image" src="https://github.com/user-attachments/assets/e17b4f58-8a11-4cde-b315-32ea6788a70e" />

### Convert Image to Grayscale

```
imageGray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(image[:,:,::-1])
plt.title("Original Image")
plt.show()
plt.imshow(imageGray,cmap='gray')
plt.title("Grayscale Image")
plt.show()
```
<img width="518" height="818" alt="image" src="https://github.com/user-attachments/assets/5f872309-df91-429d-a98f-bc8dae587de5" />

### Split Image into R,G,B Channels

```
imageB, imageG, imageR = cv2.split(image)

plt.figure(figsize=(20,12))
plt.subplot(141);plt.imshow(image[:,:,::-1]);plt.title("Original Image")
plt.subplot(142);plt.imshow(imageB,cmap='gray');plt.title("Blue Channel")
plt.subplot(143);plt.imshow(imageG,cmap='gray');plt.title("Green Channel")
plt.subplot(144);plt.imshow(imageR,cmap='gray');plt.title("Red Channel");
plt.show()

```
<img width="1411" height="399" alt="image" src="https://github.com/user-attachments/assets/34f59995-e4b5-4a37-b967-d1c92dda8c8e" />

### Perform Thresholding

```
_, thresh_inv = cv2.threshold(imageG, 20, 255, cv2.THRESH_BINARY_INV)

plt.figure(figsize=(4, 4))
plt.imshow(thresh_inv, cmap='gray')
plt.title("Threshold Binary Inverse")
plt.axis('on')  
plt.show()
```
<img width="582" height="480" alt="image" src="https://github.com/user-attachments/assets/ab1ac25a-0dfb-44a9-8789-8dee4fba7c88" />

### Perform morphological operations

```
kernel = np.ones((5, 5), dtype=np.uint8)
eroded = cv2.erode(thresh_inv,kernel,iterations=2)
plt.imshow(eroded,cmap='gray')
```

<img width="508" height="540" alt="image" src="https://github.com/user-attachments/assets/8d6bd53f-a654-4ddf-852c-be72bba76f23" />


```
dilate = cv2.dilate(thresh_inv,kernel,iterations=2)
plt.imshow(dilate,cmap='gray')
```
<img width="650" height="539" alt="image" src="https://github.com/user-attachments/assets/653a1034-36cf-48f2-86dd-f535e598a80b" />

### Create SimpleBlobDetector

```
# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0

params.minDistBetweenBlobs = 2

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia =True
params.minInertiaRatio = 0.8

# Create SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)
```

### Detect Coins

```
keypoints = detector.detect(dilate)

num_coins = len(keypoints)
print(f"Number of coins detected: {num_coins}")
plt.imshow(image[:,:,::-1]);
```
<img width="591" height="545" alt="image" src="https://github.com/user-attachments/assets/29ab7c37-5c22-4131-a21c-1ca7aba5a626" />


## OUTPUT:
The program successfully detects and counts 9 coins in the input image and each detected coin is highlighted with a red circle on the output image.
