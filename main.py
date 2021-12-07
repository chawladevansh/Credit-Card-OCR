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

# Read the OCR Reference image for both digits and characters
img = cv2.imread('ocr_ref.png')
ocra = cv2.imread('OCRA.png')
digits = processOcrRoi(img)
alphabets = processOcrRoi(ocra)

# Map every contours to a character.
char = {
    0 : ['A' , alphabets[1]],
    1 : ['B', alphabets[7]],
    2 : ['C', alphabets[12]],
    3 : ['D', alphabets[17]],
    4 : ['E', alphabets[22]],
    5 : ['F', alphabets[27]],
    6 : ['G', alphabets[32]],
    7 : ['H', alphabets[36]],
    8 : ['I', alphabets[42]],
    9 : ['J', alphabets[50]],
    10 : ['K', alphabets[54]],
    11 : ['L', alphabets[60]],
    12 : ['M', alphabets[67]],
    13 : ['N', alphabets[73]],
    14 : ['O', alphabets[82]],
    15 : ['P', alphabets[87]],
    16 : ['Q', alphabets[0]],
    17 : ['R', alphabets[6]],
    18 : ['S', alphabets[11]],
    19 : ['T', alphabets[16]],
    20 : ['U', alphabets[21]],
    21 : ['V', alphabets[26]],
    22 : ['W', alphabets[31]],
    23 : ['X', alphabets[35]],
    24 : ['Y', alphabets[41]],
    25 : ['Z', alphabets[47]],
    26 : ['A', alphabets[3]],
    27 : ['B', alphabets[9]],
    28 : ['C', alphabets[14]],
    29 : ['D', alphabets[19]],
    30 : ['E', alphabets[24]],
    31 : ['F', alphabets[29]],
    32 : ['G', alphabets[34]],
    33 : ['H', alphabets[38]],
    34 : ['I', alphabets[44]],
    35 : ['J', alphabets[49]],
    36 : ['K', alphabets[56]],
    37 : ['L', alphabets[62]],
    38 : ['M', alphabets[68]],
    39 : ['N', alphabets[74]],
    40 : ['O', alphabets[83]],
    41 : ['P', alphabets[88]],
    42 : ['Q', alphabets[2]],
    43 : ['R', alphabets[8]],
    44 : ['S', alphabets[13]],
    45 : ['T', alphabets[18]],
    46 : ['U', alphabets[23]],
    47 : ['V', alphabets[28]],
    48 : ['W', alphabets[33]],
    49 : ['X', alphabets[37]],
    50 : ['Y', alphabets[43]],
    51 : ['Z', alphabets[48]],
}

'''
 => Function to process sample credit card image

    - Smoothen the photo using multiple filters and morphological gradient transform.
    - Calculate contours of the card image.

    @return type - contours (numpy array)
'''
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

# Read the test image and call the function.
card = cv2.imread('test1.png')
conts = processCardImage(card)

# Dictionary of companies
COMPANIES = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

'''
 => Function to use template matching on the card number region of interest

    - Calculate contours on the card image 
    - Smoothen the image and use template matching to match contours from the OCR templates.
    - Reports the characters with the highest confidence value.

    @return type - string
'''
def matchCardChars(conts):
    img = cv2.imread('OCRA.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width = 300)

    locations = []

    for(i,c) in enumerate(conts):
        (x,y,w,h) = cv2.boundingRect(c)
        if y > 145 and y < (gray.shape[0]- 8) and x < (gray.shape[1] * 5 / 8) and x > 10:
            locations.append((x,y,w,h))

    locations = sorted(locations, key=lambda x:x[0])

    output = ''

    for (i, (bx, by, bw, bh)) in enumerate(locations):
        box = gray[by - 5:by + bh + 5, bx + bw + 5]
        box = cv2.threshold(box, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.waitKey(0)

        box_conts = cv2.findContours(box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        box_conts = imutils.grab_contours(box_conts)
        box_conts = contours.sort_contours(box_conts, method = 'left-to-right')[0]

        user_name = ''
        for c in box_conts:
            (x,y,w,h) = cv2.boundingRect(c)
            roi = box[y:y+h, x:x+w]
            roi = cv2.resize(roi, (57,88))

            confidence = []

            for i in range(len(char)):
                result = cv2.matchTemplate(roi, char[1][1], cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                confidence.append(score)

            i_max = np.argmax(confidence)

        output = output + " " + user_name

    return output

'''
 => Function to use template matching on the card number region of interest

    - Calculate contours on the card image and suse them to divide the number into 4 region of interests.
    - Smoothen the image and use template matching to match contours from the OCR templates.
    - Reports the character/digit with the highest confidence value.

    @return type - list
'''
def matchCardNumber(conts):
    card = cv2.imread('test1.png')
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width = 300)

    locations = []
    for(i,c) in enumerate(conts):
        (x,y,w,h) = cv2.boundingRect(c)
        
        aspect_ratio = w / float(h)
        if aspect_ratio > 2.5 and aspect_ratio < 4.0:
            if(w > 40 and w < 55) and (h > 10 and h < 20):
                    locations.append((x,y,w,h))
                    
    locations = sorted(locations, key=lambda x:x[0])

    output = []

    for (i, (bx, by, bw, bh)) in enumerate(locations):
        card_number = []
        box = gray[by - 5:by + bh + 5, bx - 5:bx + bw + 5]
        box = cv2.threshold(box, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        box_conts = cv2.findContours(box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        box_conts = imutils.grab_contours(box_conts)
        box_conts = contours.sort_contours(box_conts, method = 'left-to-right')[0]

        for c in box_conts:
            (x,y,w,h) = cv2.boundingRect(c)
            roi = box[y:y+h, x:x+w]
            roi = cv2.resize(roi, (57,88))
            confidence = []

            for(digit, d_roi) in digits.items():
                result = cv2.matchTemplate(roi, d_roi, cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                confidence.append(score)

            card_number.append(str(np.argmax(confidence)))
        
        output.extend(card_number)

    return output

number = matchCardNumber(conts)
name = matchCardChars(conts)

# print outputs
print("Card Number        : {}".format("".join(number)))
print("Card Company       : {}".format(COMPANIES[number[0]]))
print("Name               : {}".format(name))