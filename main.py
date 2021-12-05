# Import Libraries
import sys
import cv2
import imutils
import argparse
import numpy as np
from imutils import contours
import matplotlib.pyplot as plt

def processRefImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = (255-image)

    refCont = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    refCont = imutils.grab_contours(refCont)
    refCont = contours.sort_contours(refCont, method="left-to-right")[0]

    card_nums = {}

    for(digit, contour) in enumerate(refCont):
        (x, y, w, h) = cv2.boundingRect(contour)
        box = image[y:y+h, x:x+w]
        box = cv2.resize(box, (57, 88))
        card_nums[digit] = box

    return image, refCont, card_nums

img = cv2.imread('ocr_ref.png')
image, contours = processRefImage(img)

