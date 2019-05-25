import cv2
import numpy as np
import time
import sys
# np.set_printoptions(threshold=sys.maxsize)

luminance_threshold = 30
scale_percent = 30

drawing = False
ix = -1
iy = -1

alpha_slider_max = 100
trackbar_name = 'Alpha'

points_data = []
# Constraint mask
    # np.zeros = ()


def draw(event, x, y, flags, param):
    global drawing, ix, iy, points_data

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points_data = []
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (ix, iy), (x,y), (0,255,0), 3)
            ix, iy = x, y
            points_data.append([y, x])

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        constrain(np.unique(np.array(points_data), axis=0), luminance_threshold, img, luminance_channel)


def constrain(selected_points, threshold, img, lum_channel):
    assert lum_channel.shape == (img.shape[0], img.shape[1]), 'img shape and lum channel shape mismatch'

    x = selected_points[:,1]
    y = selected_points[:,0]

    selected_luminance = lum_channel[y, x]
    selected_mean = np.mean(selected_luminance)

    print('selected_points: {}').format(selected_points)
    print('selected luminance: {}'.format(selected_luminance))
    print('selected mean: {}'.format(selected_mean))

    for lx, ly in np.ndindex(luminance_channel.shape):
        # luminance_channel[ix, iy] = luminance value at pixel (x, y)
        if (np.absolute(selected_mean - luminance_channel[lx, ly]) < threshold):
            # How to apply constraint weights?
            img[lx, ly] = (luminance_channel[lx, ly],0,0)

            # THINGS TO CHECK:
                # xy coordinate correct?
                # is the pen color affecting?

#def gaussian_falloff(mean_luminance, threshold, img):

def on_trackbar(val):
    alpha = val / alpha_slider_max
    beta = ( 1.0 - alpha )
    #dst = cv2.addWeighted(im_rgb1, alpha, im_rgb2, beta, 0.0)
    #cv2.imshow('itm', dst)

def resize_img(img, scale_pct):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized

if __name__ == '__main__':
    im_rgb = cv2.imread('img/test3.jpg')
    # Conver to CIE LAB space
    img = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2Lab)

    im_rgb = resize_img(im_rgb, scale_percent)
    img = resize_img(img, scale_percent)
    luminance_channel, a, b = cv2.split(img)

    print('img shape (y,x): {}'.format(im_rgb.shape))

    cv2.namedWindow('itm')
    cv2.setMouseCallback('itm', draw)

    cv2.namedWindow('ref')
    cv2.imshow('ref', im_rgb)

    # Trackbar doesn't do anything for now
    #cv2.createTrackbar(trackbar_name, 'itm', 0, alpha_slider_max, on_trackbar)
    #on_trackbar(0)

    while(1):
        cv2.imshow('itm', img)
        k=cv2.waitKey(1)&0xFF
        if k==27:
            break
    cv2.destroyAllWindows()



