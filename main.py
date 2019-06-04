import cv2
import numpy as np
import time
import sys

constraint_threshold = 20
scale_percent = 15

drawing = False
ix = -1
iy = -1

points_data = []

def draw(event, x, y, flags, param):
    global drawing, ix, iy, points_data

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points_data = []
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (ix, iy), (x,y), (0,255,0), 6)
            ix, iy = x, y
            points_data.append([y, x])

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        constrain(np.unique(np.array(points_data), axis=0), constraint_threshold)

def constrain(selected_points, threshold):
    assert luminance_channel.shape == (img.shape[0], img.shape[1]), 'img shape and lum channel shape mismatch'
    assert luminance_channel.shape == constraint_mask.shape, 'constraint mask and lum channel shape mismatch'

    transparent_constraint = np.zeros(img.shape)

    x = selected_points[:,1]
    y = selected_points[:,0]

    selected_luminance = luminance_channel[y, x]
    selected_a = a_channel[y,x]
    selected_b = b_channel[y,x]

    luminance_mean = np.mean(selected_luminance)
    a_mean = np.mean(selected_a)
    b_mean = np.mean(selected_b)

    print('selected_points: {}'.format(selected_points))
    print('selected luminance: {}'.format(selected_luminance))
    print('luminance mean: {}'.format(luminance_mean))

    lischinski_minimization(3,3)

    #Constrain brushed pixels
    constraint_mask[y,x] = 1

    for y, x in zip(y,x):
        #img[y,x] = (luminance_channel[y,x], 0, 0)
        img_rgb[y, x] = (luminance_channel[y, x], 0, 0)

    # Lumachrome brush
    for lx, ly in np.ndindex(luminance_channel.shape):
        if np.absolute(luminance_mean - luminance_channel[lx, ly]) < threshold:
            if np.sqrt(((a_mean-a_channel[lx,ly]) **2) + ((b_mean-b_channel[lx,ly]) **2)) < threshold:
                # Showing constrained pixels
                #img[lx, ly] = (luminance_channel[lx, ly],0,0)
                img_rgb[lx, ly] = (255, 0, 100)
                # Add constrained pixels from the brush propagation
                constraint_mask[lx, ly] = 1
                cv2.imshow('ref', img_rgb)


def lischinski_minimization(g, LM_alpha =1, LM_lambda = 0.2, LM_epsilon = 0.0001):
    # g = target exposure value

    # log-luminance channel
    L = np.log(luminance_channel)

    # Build b vector
    b = g * constraint_mask

    neighbors = get_neighboring_luminance(L, 3, 3)
    A = []

    for ly, lx in np.ndindex(L.shape):
        L_neighbors = get_neighboring_luminance(L, ly, lx)
        a = LM_lambda / (np.power(np.abs(L[ly,lx] - L_neighbors), LM_alpha) + LM_epsilon)



def get_neighboring_luminance(L, y,x):
    j = []

    j.append(L[y,x - 1])
    j.append(L[y, x + 1])
    j.append(L[y+1, x])
    j.append(L[y-1, x])

    return np.array(j)

def on_trackbar(val):
    alpha = 1
    beta = val
    new_image = np.zeros(img_rgb.shape, img_rgb.dtype)

    for y in range(img_rgb.shape[0]):
        for x in range(img_rgb.shape[1]):
            for c in range(img_rgb.shape[2]):
                new_image[y, x, c] = np.clip(alpha * img_rgb[y, x, c] + beta, 0, 255)
    cv2.imshow('output', new_image)


def resize_img(img, scale_pct):
    width = int(img.shape[1] * scale_pct / 100)
    height = int(img.shape[0] * scale_pct / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized

if __name__ == '__main__':
    img_rgb = cv2.imread('img/test3.jpg')
    # Conver to CIE LAB space
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)

    img_rgb = resize_img(img_rgb, scale_percent)
    img = resize_img(img, scale_percent)
    luminance_channel, a_channel, b_channel = cv2.split(img)

    print('img shape (y,x): {}'.format(img_rgb.shape))
    # Mask containing weight of each constrained pixel
    constraint_mask = np.zeros((img.shape[0], img.shape[1]))

    cv2.namedWindow('itm')
    cv2.setMouseCallback('itm', draw)

    cv2.namedWindow('ref')
    cv2.imshow('ref', img_rgb)

    cv2.namedWindow('output')
    cv2.imshow('output', img_rgb)


    # Trackbar doesn't do anything for now
    cv2.createTrackbar('Brightness', 'itm', 10, 50, on_trackbar)
    on_trackbar(0)

    while(1):
        cv2.imshow('itm', img)
        k=cv2.waitKey(1)&0xFF
        if k==27:
            break
    cv2.destroyAllWindows()
