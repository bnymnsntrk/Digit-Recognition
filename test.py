# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
from PIL import Image

# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Read the input image
im = cv2.imread("image/sample.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow("grayscale", im_gray)
cv2.waitKey()
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
cv2.imshow("blurred", im_gray)
cv2.waitKey()

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("threshold", im_th)
cv2.waitKey()

# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
print(rects)

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
i = 1
for rect in rects:
    print(i)
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    len = int(rect[3] * 1.6)
    print(len)
    pt1 = int(rect[1] + rect[3] // 2 - len // 2)
    pt2 = int(rect[0] + rect[2] // 2 - len // 2)

    if pt1 < 0:
        pt1 = 0
    if pt2 < 0:
        pt2 = 0

    print(pt1, pt2)
    roi = im_th[pt1:pt1+len, pt2:pt2+len]

    cv2.imshow("roi", roi)
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm='L1',
                     visualise=None)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    i += 1

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()