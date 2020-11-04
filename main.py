
if __name__ == '__main__':
    import cv2
    import numpy as np


    def onChange(temp):
        pass


    def trackBar():
        image = cv2.imread('1.jpg', cv2.COLOR_BGR2GRAY)
        cv2.namedWindow('Track Bar')
        cv2.resizeWindow('Track Bar', 800, 800)

        interp = cv2.INTER_AREA

        cv2.createTrackbar('Scale', 'Track Bar', 1, 4, onChange)
        cv2.createTrackbar('Iteration', 'Track Bar', 1, 20, onChange)
        cv2.createTrackbar('Kernel', 'Track Bar', 1, 10, onChange)
        cv2.createTrackbar('File', 'Track Bar', 1, 6, onChange)
        cv2.createTrackbar('Mode', 'Track Bar', 1, 2, onChange)
        now_file = 1

        while True:
            scale = cv2.getTrackbarPos('Scale', 'Track Bar')
            iter = cv2.getTrackbarPos('Iteration', 'Track Bar')
            kernel_size = cv2.getTrackbarPos('Kernel', 'Track Bar')
            file = cv2.getTrackbarPos('File', 'Track Bar')
            switch = cv2.getTrackbarPos('Mode', 'Track Bar')

            if file != now_file:
                image = cv2.imread(str(file) + '.jpg', cv2.COLOR_BGR2GRAY)
                if file in [0, 4, 5, 6]:
                    image = cv2.imread(str(file) + '.png', cv2.COLOR_BGR2GRAY)
                now_file = file

            img = image.copy()

            if file in [1,2]:
                _, thr = cv2.threshold(img, img[0][0] + 10, 255, cv2.THRESH_BINARY)
                img = cv2.ximgproc.thinning(thr)

            ############################################################################
            if switch == 1:
                if scale <= 0:
                    scale = 1

                scale = 1/scale

                resize = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interp)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

                res = cv2.morphologyEx(resize, cv2.MORPH_OPEN, kernel, iterations=iter)
                res = cv2.resize(res, (0, 0), fx=1 / scale, fy=1 / scale, interpolation=interp)
                cv2.imwrite('test' + str(file) + '.jpg', res)
                res = cv2.resize(res, (800, 800))
                cv2.imshow('Track Bar', res)
                cv2.waitKey()
            ############################################################################
            else:
                # contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                kernel = np.ones((3,3), np.int8)

                dilate_img = img.copy()
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                dilate_img = cv2.dilate(dilate_img, kernel, iterations=1)

                contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                index = 0
                print(hierarchy)
                for b in contours:

                    if hierarchy[0][index][2] != -1:
                        index += 1
                        continue
                    index += 1
                    x_pos, y_pos = b[0][0]

                    print("x: {}, y: {}".format(x_pos, y_pos))

                    # cv2.circle(img, (x_pos, y_pos), 7, (0, 255, 0), 1)
                    cv2.floodFill(img, None, (x_pos, y_pos), (255, 0, 200))
                # contours, _ = cv2.findContours(img, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
                # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # img = cv2.drawContours(img, contours, -1, [0,200,0], -1)
                # img = cv2.resize(img, (800, 800))
                cv2.imshow('Track Bar', img)
                cv2.imwrite('test' + str(file) + '.jpg', img)
                cv2.waitKey()
            ############################################################################

    trackBar()