import cv2
import numpy as np

drawing = False
ix = -1
iy = -1

def draw(event, x, y, flags, param):
    global drawing, ix, iy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(im, (ix, iy), (x,y), (0,255,0), 5)
            ix = x
            iy = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    return x, y

im = cv2.imread("test.jpg")
cv2.namedWindow("Interactive local tone adjustment")
cv2.setMouseCallback("Interactive local tone adjustment", draw)

while(1):
    cv2.imshow('Interactive local tone adjustment', im)
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break
cv2.destroyAllWindows()






