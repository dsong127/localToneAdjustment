import cv2
import numpy as np
from scipy.sparse import diags

luminance_threshold = 30
chroma_threshold = 5
scale_percent = 20
beta = 30
alpha = 2

drawing = False
ix = -1
iy = -1

points_data = []

def draw(event, x, y, flags, param):
    global drawing, ix, iy, points_data, begin, end

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points_data = []
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            ix, iy = x, y
            cv2.line(img_rgb, (ix, iy), (x, y), (0, 255, 0), 5)
            points_data.append([y, x])

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        constrain(np.unique(np.array(points_data), axis=0))

def constrain(selected_points):
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

    # Regular brush
    # for y, x in zip(y,x):
    #     img_rgb[y, x] = (luminance_channel[y, x], 0, 0)

    # Luminance brush
    for lx, ly in np.ndindex(luminance_channel.shape):
        if np.absolute(luminance_mean - luminance_channel[lx, ly]) < luminance_threshold:
            # Lumachrome brush
            #if np.sqrt(((a_mean-a_channel[lx,ly]) **2) + ((b_mean-b_channel[lx,ly]) **2)) < chroma_threshold:
                # Showing constrained pixels
            img_rgb[lx, ly] = (255, 0, 100)
            img_output[lx,ly] = np.clip(alpha * img_output[lx,ly] + beta,0,255)
                # Add constrained pixels from the brush propagation
            constraint_mask[lx, ly] = 1
            cv2.imshow('output', img_output)

    # change_brightness(result)

def minimizer(g, LM_alpha =1, LM_lambda = 0.2, LM_epsilon = 0.0001):
    # g = target exposure value

    # log-luminance channel
    L = np.log(luminance_channel)
    size = luminance_channel.shape
    r = size[0]
    c = size[1]
    n = r*c

    # Build b vector
    g = g * constraint_mask
    b = g.reshape(n, 1)

    # Compute gradients
    dy = np.diff(L, n = 1, axis = 0)
    dy = - LM_lambda / np.abs(np.power(dy, LM_alpha) + LM_epsilon)
    dy = np.pad(dy, ((0,1),(0,0)), 'constant')
    # Reshape to column vector
    dy = dy.reshape(dy.shape[0]*dy.shape[1], order='F')
    dy = dy.reshape((dy.size,1))

    dx = np.diff(L, n=1, axis=1)
    dx = - LM_lambda / np.abs(np.power(dx, LM_alpha) + LM_epsilon)
    dx = np.pad(dx, ((0,0), (0,1)), 'constant')
    dx = dx.reshape(dx.shape[0] * dx.shape[1], order='F')
    dx = dx.reshape((dx.size, 1))

    # Build A
    A = np.spdiags([dx, dy], [-r, -1], n, n);
    A = diags([dx, dy], [-r,-1], shape= (n,n)).toarray()
    A = A + A.T

    g00 = np.pad(dx, r, 'constant')
    g01 = padarray(dy, 1, 'constant')
    D = W.reshape(r * c,1) - (g00 + dx + g01 + dy)
    A = A + np.spdiags(D, 0, n, n);

    result, L = cv2.solve(A, b, flags = cv2.DECOMP_SVD)

    return result

def on_trackbar(val):
    alpha = 1
    beta = val
    new_image = np.zeros(img_rgb.shape, img_rgb.dtype)

    for y in range(img_rgb.shape[0]):
        for x in range(img_rgb.shape[1]):
            for c in range(img_rgb.shape[2]):
                new_image[y, x, c] = np.clip(alpha * img_output[y, x, c] + beta, 0, 255)
    cv2.imshow('output', new_image)


def resize_img(img, scale_pct):
    width = int(img.shape[1] * scale_pct / 100)
    height = int(img.shape[0] * scale_pct / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized

if __name__ == '__main__':
    img_rgb = cv2.imread('img/test1.jpg')
    # Convert to CIE LAB space
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)

    img_rgb = resize_img(img_rgb, scale_percent)

    img_output = cv2.imread('img/test1.jpg')
    img_output = resize_img(img_output, scale_percent)

    img = resize_img(img, scale_percent)
    luminance_channel, a_channel, b_channel = cv2.split(img)

    print('img shape (y,x): {}'.format(img_rgb.shape))
    # Matrix containing weight of each constrained pixel
    constraint_mask = np.zeros((img.shape[0], img.shape[1]))

    cv2.namedWindow('ref')
    cv2.setMouseCallback('ref', draw)

    cv2.namedWindow('output')
    cv2.imshow('output', img_output)

    # cv2.createTrackbar('Brightness', 'itm', 10, 50, on_trackbar)
    # on_trackbar(0)

    while(1):
        cv2.imshow('ref', img_rgb)
        k=cv2.waitKey(1)&0xFF
        if k==27:
            break
    cv2.destroyAllWindows()