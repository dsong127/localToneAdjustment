import cv2
import numpy as np
from scipy.sparse import diags
import time
import sys

constraint_threshold = 15
scale_percent = 20

drawing = False
ix = -1
iy = -1
begin = -1
end = -1

points_data = []

# ret, L = cv2.solve(A, b, flags = cv2.DECOMP_SVD)

def draw(event, x, y, flags, param):
    global drawing, ix, iy, points_data, begin, end

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points_data = []
        ix, iy = x, y
        begin = x
        end = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            ix, iy = x, y
            cv2.line(img_rgb, (ix, iy), (x, y), (0, 255, 0), 5)
            points_data.append([y, x])

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        constrain(np.unique(np.array(points_data), axis=0), constraint_threshold)
        #cv2.line(img_rgb, (begin, end), (x, y), (0, 255, 0), 5)


def constrain(selected_points, threshold):
    assert luminance_channel.shape == (img.shape[0], img.shape[1]), 'img shape and lum channel shape mismatch'
    assert luminance_channel.shape == constraint_mask.shape, 'constraint mask and lum channel shape mismatch'

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

    #Constrain brushed pixels
    constraint_mask[y,x] = 1

    # for y, x in zip(y,x):
    #     img_rgb[y, x] = (luminance_channel[y, x], 0, 0)

    # Lumachrome brush
    for lx, ly in np.ndindex(luminance_channel.shape):
        if np.absolute(luminance_mean - luminance_channel[lx, ly]) < threshold:
            #if np.sqrt(((a_mean-a_channel[lx,ly]) **2) + ((b_mean-b_channel[lx,ly]) **2)) < threshold:
                # Showing constrained pixels
            img_rgb[lx, ly] = (255, 0, 100)
                # Add constrained pixels from the brush propagation
            constraint_mask[lx, ly] = 1
            cv2.imshow('ref', img_rgb)

def lischinski_minimization(g, LM_alpha =1, LM_lambda = 0.2, LM_epsilon = 0.0001):
    # g = target exposure value

    # log-luminance channel
    L = np.log(luminance_channel)
    size = luminance_channel.shape
    r = size[0]
    c = size[1]
    n = r*c

    # Build b vector
    g = g * constraint_mask
    b = g.reshape(n, 1) #(79544,1)  ---> CHECK DIMENSIONS

    # Compute gradients
    dy = np.diff(L, n = 1, axis = 0)
    #print(dy.shape) # 325, 244
    dy = - LM_lambda / np.abs(np.power(dy, LM_alpha) + LM_epsilon)
    #print(dy.shape) # 325, 244
    dy = np.pad(dy, ((0,1),(0,0)), 'constant')
    #print(dy.shape) # 326, 245
    # Reshape to column vector
    dy = dy.reshape(dy.shape[0]*dy.shape[1], order='F')
    dy = dy.reshape((dy.size,1))

    dx = np.diff(L, n=1, axis=1)
    dx = - LM_lambda / np.abs(np.power(dx, LM_alpha) + LM_epsilon)
    dx = np.pad(dx, ((0,0), (0,1)), 'constant')
    dx = dx.reshape(dx.shape[0] * dx.shape[1], order='F')
    dx = dx.reshape((dx.size, 1))

    print(dy.shape)
    print(dx.shape)

    # Build A
    #A = np.spdiags([dx, dy], [-r, -1], n, n);
    A = diags([dx, dy], [-r,-1], shape= (n,n)).toarray()
    A = A + A.T


    # A = np.zeros(img_rgb.shape)
    # for ly, lx in np.ndindex(L.shape):
    #     #If j is neighboring pixel
    #         # -lambda / gradient
    #     #else if weight - Aik --> What is Aik?
    #     #else 0
    #     L_neighbors = get_neighboring_luminance(L, ly, lx)
    #     A[ly,lx] = -LM_lambda / (np.power(np.abs(L[ly,lx] - L_neighbors), LM_alpha) + LM_epsilon)

#def is_neighboring_pixel(y,x):
    # input j = (y, x)
    #

# def get_neighboring_luminance(L, y,x):
#     j = []
#
#     # 4 neighboring pixels of i.
#     j.append(L[y,x - 1])
#     j.append(L[y, x + 1])
#     j.append(L[y+1, x])
#     j.append(L[y-1, x])
#
#     return np.array(j)

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
    img_rgb = cv2.imread('img/test8.jpg')
    # Convert to CIE LAB space
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)

    img_rgb = resize_img(img_rgb, scale_percent)
    img_output = img_rgb
    img = resize_img(img, scale_percent)
    luminance_channel, a_channel, b_channel = cv2.split(img)

    print('img shape (y,x): {}'.format(img_rgb.shape))
    # Matrix containing weight of each constrained pixel
    constraint_mask = np.zeros((img.shape[0], img.shape[1]))

    cv2.namedWindow('ref')
    cv2.setMouseCallback('ref', draw)

    # cv2.namedWindow('output')
    # cv2.imshow('output', img_output)

    #lischinski_minimization(3)

    # This is really slow
    # cv2.createTrackbar('Brightness', 'itm', 10, 50, on_trackbar)
    # on_trackbar(0)

    while(1):
        cv2.imshow('ref', img_rgb)
        k=cv2.waitKey(1)&0xFF
        if k==27:
            break
    cv2.destroyAllWindows()