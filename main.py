
if __name__ == '__main__':
    import cv2
    import numpy as np


    def onChange(temp):
        return temp


    def trackBar():
        image = cv2.imread('1.jpg', cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (800, 800))
        cv2.namedWindow('Track Bar')
        cv2.resizeWindow('Track Bar', 800, 800)

        interp = cv2.INTER_AREA

        cv2.createTrackbar('Scale', 'Track Bar', 1, 4, onChange)
        cv2.createTrackbar('Iteration', 'Track Bar', 1, 20, onChange)
        cv2.createTrackbar('Kernel', 'Track Bar', 1, 10, onChange)
        cv2.createTrackbar('File', 'Track Bar', 1, 3, onChange)

        now_file = 1
        while True:
            scale = cv2.getTrackbarPos('Scale', 'Track Bar')
            iter = cv2.getTrackbarPos('Iteration', 'Track Bar')
            kernel_size = cv2.getTrackbarPos('Kernel', 'Track Bar')
            file = cv2.getTrackbarPos('File', 'Track Bar')

            if file != now_file:
                image = cv2.imread(str(file) + '.jpg', cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (800, 800))
                now_file = file

            img = image.copy()

            if scale <= 0:
                scale = 1

            resize = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interp)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            res = cv2.morphologyEx(resize, cv2.MORPH_CLOSE, kernel, iterations=iter)
            res = cv2.resize(res, (0, 0), fx=1 / scale, fy=1 / scale, interpolation=interp)

            cv2.imshow('Track Bar', res)
            k = cv2.waitKey(1) & 0xFF

            if k == 27:
                break

    trackBar()