# Import Libraries
import sys
import cv2
import imutils
import argparse
import numpy as np
from imutils import contours
import matplotlib.pyplot as plt

def resize_image(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized

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
image, contours, digits = processRefImage(img)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, card = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
# # Credit Card Image Processing
# card = cv2.imread('test1.png')
    card = resize_image(card, width = 300)
    card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

    kernel93 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    # Tophat morphological transformation
    tophat = cv2.morphologyEx(card_gray, cv2.MORPH_TOPHAT, kernel93)

    # Sobel Filter gradient smoothening over x Axis
    sobel = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    sobel = np.absolute(sobel)

    (minThresh, maxThresh) = (np.min(sobel), np.max(sobel))

    sobel = (255 * ((sobel - minThresh) / (maxThresh - minThresh)))
    sobel = sobel.astype("uint8")

    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    sobel = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, kernel93)
    sobel_thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    sobel_thresh = cv2.morphologyEx(sobel_thresh, cv2.MORPH_CLOSE, kernel5)

    # Credit Card Image Contours
    card_contours = cv2.findContours(sobel_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_contours = imutils.grab_contours(card_contours)
    digit_locations = []

    for (digit, contour) in enumerate(card_contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if aspect_ratio > 2.5 and aspect_ratio < 4.0:
            if(w > 40 and w < 55) and (h > 10 and h < 20):
                digit_locations.append((x,y,w,h))

    digit_locations = sorted(digit_locations, key=lambda x:x[0])


    #####
    output = []

    for (i, (X, Y, W, H)) in enumerate(digit_locations):
        groupOp = []
        group = card_gray[Y - 5:Y + H + 5, X - 5:X + W + 5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        digitContours = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digitContours = imutils.grab_contours(digitContours)
        #digitContours = contours.sort_contours(digitContours, method="left-to-right")[0]

        for contour in digitContours:
            (x,y,w,h) = cv2.boundingRect(contour)
            point = group[y:y+h, x:x+w]
            point = cv2.resize(point, (57,88)) ##########

            matches = []
            for(digit, points) in digits.items():
                # Template matching
                output = cv2.matchTemplate(point, points, cv2.TM_CCOEFF)
                (_, match, _, _) = cv2.minMaxLoc(output)
                matches.append(match)

                groupOp.append(str(np.argmax(matches)))

                cv2.rectangle(card, (X - 5, Y - 5), (X + W + 5, Y + H + 5), (0, 0, 255), 2)
                cv2.putText(card, "".join(groupOp), (X, Y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                #output.extend(groupOp)

    # cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
    # plt.imshow(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
    # plt.title('Image'); plt.show()

    cv2.imshow('frame', card)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()