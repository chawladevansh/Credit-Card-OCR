import cv2
import imutils
from imutils import contours
import numpy as np

'''
 => Function to process the OCR reference image

    - Calculate contours of the reference image for each digit
    - Store the ROI of every digit in a dictionary

    @return type - dict
'''
def processOcrRoi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

    conts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = contours.sort_contours(conts, method = "left-to-right")[0]

    digits = {}

    for(i, c) in enumerate(conts):
        (x,y,w,h) = cv2.boundingRect(c)
        roi = thresh[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57, 88))
        digits[i] = roi

    return digits

img = cv2.imread('ocr_ref.png')
digits = processOcrRoi(img)

def processCardImage(img):
    
    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width = 300)

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectkernel)

    sobel = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    sobel = np.absolute(sobel)
    (minThresh, maxThresh) = (np.min(sobel), np.max(sobel))

    sobel = 255 * (sobel - minThresh) / (maxThresh - minThresh)
    sobel = sobel.astype('uint8')

    sobel = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, rectkernel)

    thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqkernel)

    conts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = contours.sort_contours(conts, method='left-to-right')[0]

    return conts

card = cv2.imread('test1.png')
conts = processCardImage(card)