import cv2
import numpy as np 

WIDTH, HEIGHT = 3000,4000

def scan_detection(image):
    doc_contour = np.array([[0,0], [WIDTH,0], [WIDTH,HEIGHT], [0,HEIGHT]])

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5,5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                doc_contour = approx
                max_area = area
    
    cv2.drawContours(frame, [doc_contour], -1, (0,255,0), 3)
    