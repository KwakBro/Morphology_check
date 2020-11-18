import numpy as np


if __name__ == '__main__':
    import cv2
    import numpy as np


    def onChange(temp):
        pass


    def trackBar():

        cv2.namedWindow('Track Bar')
        cv2.resizeWindow('Track Bar', 800, 800)

        interp = cv2.INTER_AREA

        cv2.createTrackbar('maxGap', 'Track Bar', 1, 50, onChange)
        cv2.createTrackbar('Threshold', 'Track Bar', 1, 200, onChange)
        cv2.createTrackbar('minLen', 'Track Bar', 1, 50, onChange)
        cv2.createTrackbar('File', 'Track Bar', 1, 16, onChange)
        cv2.createTrackbar('Mode', 'Track Bar', 1, 3, onChange)

        while True:
            scale = cv2.getTrackbarPos('maxGap', 'Track Bar')
            iter = cv2.getTrackbarPos('Threshold', 'Track Bar')
            kernel_size = cv2.getTrackbarPos('minLen', 'Track Bar')
            file = cv2.getTrackbarPos('File', 'Track Bar')
            switch = cv2.getTrackbarPos('Mode', 'Track Bar')

            if file == 0:
                herris = cv2.imread('w' + str(file + 1) + '.png', cv2.IMREAD_GRAYSCALE)
                gray = np.float32(herris)
                dst = cv2.cornerHarris(gray, scale, kernel_size, iter/100)
                dst = cv2.dilate(dst, None)

                _, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, cv2.THRESH_BINARY)
                dst = np.uint8(dst)

                _, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
                cornerPoint = []
                for c in corners:
                    x, y = c
                    cornerPoint.append([int(round(x)), int(round(y))])

                for corner in cornerPoint:
                    x, y = corner
                    cv2.circle(herris, (x, y), 5, 100, -1)

                cv2.imshow('Track Bar', herris)
                cv2.waitKey()

            else:
                image = cv2.imread('w' + str(file + 1) + '.jpg')
                imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                _, thr = cv2.threshold(imgray, imgray[0][0]+5, 255, cv2.THRESH_BINARY)

                thr = cv2.ximgproc.thinning(thr)
                edges = cv2.Canny(thr, 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(edges, 1, 3.1452/180, threshold=iter,
                                        minLineLength=kernel_size, maxLineGap=scale)

                imgray = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)
                cv2.imwrite('skeleton.jpg', thr)
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(imgray, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    w = x1 - x2
                    h = y1 - y2
                    bit = [-h / (abs(w) + abs(h)), w / (abs(w) + abs(h))]
                    color = (255, 100, 0)
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    for i in range(4,  70):
                        try:
                            calc_x = int(center_x + bit[0] * i)
                            calc_y = int(center_y + bit[1] * i)

                            if thr[calc_y][calc_x] != 0:
                                cv2.line(imgray, (center_x, center_y),
                                                 (calc_x, calc_y), color, 2)
                                pts1 = np.array([[x1, y1], [x2, y2],
                                                [x2 + int(bit[0] * i), y2 + int(bit[1] * i)],
                                                [x1 + int(bit[0] * i), y1 + int(bit[1] * i)]])
                                cv2.fillPoly(imgray, [pts1], color, cv2.LINE_AA)
                                # result = i
                                # print(f'Hough Distance = {result}')
                                break

                            calc_xm = int(center_x - bit[0] * i)
                            calc_ym = int(center_y - bit[1] * i)

                            if thr[calc_ym][calc_xm] != 0:
                                cv2.line(imgray, (center_x, center_y),
                                                 (calc_xm, calc_ym), color, 2)
                                pts2 = np.array([[x1, y1], [x2, y2],
                                                 [x2 - int(bit[0] * i), y2 - int(bit[1] * i)],
                                                 [x1 - int(bit[0] * i), y1 - int(bit[1] * i)]])
                                cv2.fillPoly(imgray, [pts2], (0, 255, 0), cv2.LINE_AA)
                                # result = i
                                # print(f'Hough Distance = {result}')
                                break

                            # cv2.line(imgray, (center_x, center_y),
                            #          (calc_x, calc_y), (255, 0, 255), 1)
                            # cv2.line(imgray, (center_x, center_y),
                            #          (calc_xm, calc_ym), (255, 0, 255), 1)
                        except IndexError:
                            continue
                print("\n ---------------------------- \n")
                imgray = cv2.cvtColor(imgray, cv2.COLOR_BGR2GRAY)
                _, imgray = cv2.threshold(imgray, imgray[0][0] + 5, 255, cv2.THRESH_BINARY)

                iter = 4
                kernel_size = 5

                # 1. MorphologyEx Closing
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                imgray = cv2.morphologyEx(imgray, cv2.MORPH_CLOSE, kernel, iterations=iter)

                # 2. Eroding
                kernel = np.ones((3, 3), np.int8)
                imgray = cv2.erode(imgray, kernel, iterations=4)

                cv2.imshow('Track Bar', imgray)
                cv2.waitKey()

            # TODO : Case 1
            '''
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            res = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=iter)

            _, res = cv2.threshold(res, res[0][0] + 1, 255, cv2.THRESH_BINARY)

            res = cv2.resize(res, None, fx=1/2, fy=1/2)
            cv2.imshow('Track Bar', imgray)
            cv2.waitKey()
            '''

            # TODO : Case 2
            '''
            # if file in [1,2]:
            #     _, thr = cv2.threshold(img, img[0][0] + 10, 255, cv2.THRESH_BINARY)
            #     img = cv2.ximgproc.thinning(thr)

            ############################################################################
            if switch == 3:
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
            elif switch == 3:
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
            else:
                pixel_range = cv2.getTrackbarPos('Scale', 'Track Bar')
                Harris_R = cv2.getTrackbarPos('Iteration', 'Track Bar')
                sobel = cv2.getTrackbarPos('Kernel', 'Track Bar')

                gray = np.float32(img)
                dst = cv2.cornerHarris(gray, pixel_range, sobel, Harris_R/100)
                dst = cv2.dilate(dst, None)

                _, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, cv2.THRESH_BINARY)
                dst = np.uint8(dst)

                _, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

                for c in corners:
                    x, y = c
                    cv2.circle(img, (int(round(x)), int(round(y))), 3, 200, -1)

                cv2.imshow('Track Bar', img)
                cv2.imwrite('test' + str(file) + '.jpg', img)
                cv2.waitKey()
            '''
    trackBar()