import cv2
import numpy as np

luminance_threshold = 100
drawing = False
ix = -1
iy = -1

# test img size = (350, 950, 3)

points_data = []

def draw(event, x, y, flags, param):
    global drawing, ix, iy, points_data

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points_data = []
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(im_rgb, (ix, iy), (x,y), (0,255,0), 3)
            ix, iy = x, y
            points_data.append([x, y])

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        constrain(np.unique(np.array(points_data), axis=0), luminance_threshold, img)

    # return points_data

def constrain(selected_points, threshold, img):
    luminance_channel, a, b = cv2.split(img)
    print(luminance_channel)
    # print(selected_points)

    selected_luminance = luminance_channel[selected_points]
    selected_mean = np.mean(selected_luminance)

    print(selected_luminance)
    print(selected_mean)

    #print(luminance_channel.shape) #(350, 950)

    for lx, ly in np.ndindex(luminance_channel.shape):
        # luminance_channel[ix, iy] = l at pixel x, y
        if (np.absolute(selected_mean - luminance_channel[lx, ly]) > threshold):
            im_rgb[lx, ly] = (0,0,0)


if __name__ == '__main__':
    im_rgb = cv2.imread('img/test1.jpg')
    img = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2Lab)
    cv2.namedWindow('itm')
    cv2.setMouseCallback('itm', draw)

    while(1):
        cv2.imshow('itm', im_rgb)
        k=cv2.waitKey(1)&0xFF
        if k==27:
            break
    cv2.destroyAllWindows()



