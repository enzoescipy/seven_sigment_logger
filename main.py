from codecs import escape_encode
import enum
import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils import contours as ctr
import imutils
from imutils.perspective import four_point_transform
plt.style.use('dark_background')

class SegmentProcessor:
    IMG_RESIZE = 1000
    BLUR_BIAS = 1
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 0, 1): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }
    REVERSE_COLOR = True
    THERSH_HOLD_GLOBAL = False

    rectpos = []
    @classmethod
    def img_change(cls, image) -> None:
        cls.image = imutils.resize(image, height=(cls.IMG_RESIZE))


    
    @classmethod
    def lcd_rect_setter(cls):
        img = cls.image
        #user select drawing range by clicking 4-pointed lcd rect.
        def on_mouse(event, x, y, flag, param):
            if event == cv2.EVENT_LBUTTONDOWN :
                cls.rectpos.append([[x,y]])
                cv2.circle(img, (x,y), 3, (0,255,0), -1)
                cv2.imshow("image", img)
                if len(cls.rectpos) == 4:
                    cv2.destroyAllWindows()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_mouse, img)


        cv2.imshow("image", img)
        cv2.waitKey(0)
    @classmethod
    def segment_getter(cls):
        img = cls.image
        img = four_point_transform(img, np.array(cls.rectpos).reshape(4,2))

        #image preprocessing
        height, width, channel = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=cls.BLUR_BIAS)

        if cls.THERSH_HOLD_GLOBAL == False:
            img_blur_thresh = cv2.threshold(img_blurred, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        else:
            img_blur_thresh = cv2.adaptiveThreshold(
                img_blurred,
                maxValue=255.0,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY_INV,
                blockSize=19,
                C=9
            )


        if cls.REVERSE_COLOR:
            img_blur_thresh = 255 - img_blur_thresh

        cv2.imshow("check", img_blur_thresh)
        cv2.waitKey(0)

        #contours making

        contours, _ = cv2.findContours(
            img_blur_thresh,
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_SIMPLE
        )

        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255,255,255))



        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        contours_dict = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(temp_result, pt1=(x,y), pt2=(x+w, y+h), color=(255,255,255), thickness=2)
            
            contours_dict.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
            })
            





        #contours selecting

        possible_contours = []


        cnt = 0
        for d in contours_dict:
            area = d['w'] * d['h']
            ratio = d['w'] / d['h']
            
            if d['h'] > height*0.7:
                d['idx'] = cnt
                cnt += 1
                possible_contours.append(d['contour'])

        temp_result = np.zeros((height, width, channel), dtype = np.uint8)

        digitCnts = ctr.sort_contours(possible_contours,
            method="left-to-right")[0]
        digits = []

        # loop over each of the digits
        for c in digitCnts:
            # extract the digit ROI
            (x, y, w, h) = cv2.boundingRect(c)
            reduce = 20
            roi = img_blur_thresh[y:y + h, x:x + w]
            # compute the width and height of each of the 7 segments
            # we are going to examine
            (roiH, roiW) = roi.shape
            (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
            dHC = int(roiH * 0.05)
            # define the set of 7 segments
            reduce_start = 0.33
            reduce_end = 0.66
            segments = [
                ((int(w*reduce_start), 0), (int(w*reduce_end), dH)),	# top
                ((0, int((h // 2)*reduce_start)), (dW, int((h // 2)*reduce_end))),	# top-left
                ((w - dW, int((h // 2)*reduce_start)), (w, int((h // 2)*reduce_end))),	# top-right
                ((int(w*reduce_start), (h // 2) - dHC) , (int(w*reduce_end), (h // 2) + dHC)), # center
                ((0, (h // 2)+int((h // 2)*reduce_start)), (dW, (h // 2)+ int((h // 2)*reduce_end))),	# bottom-left
                ((w - dW, (h // 2)+int((h // 2)*reduce_start)), (w, (h // 2)+ int((h // 2)*reduce_end))),	# bottom-right
                ((int(w*reduce_start), h - dH), (int(w*reduce_end), h))	# bottom
            ]
            on = [0] * len(segments)
            # loop over the segments
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                # extract the segment ROI, count the total number of
                # thresholded pixels in the segment, and then compute
                # the area of the segment
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)
                # if the total number of non-zero pixels is greater than
                # 50% of the area, mark the segment as "on"
                if total / float(area) > 0.5:
                    on[i]= 1
            # lookup the digit and draw it on the image
            if w < 15:
                digit = 1
            elif tuple(on) in cls.DIGITS_LOOKUP.keys():
                digit = cls.DIGITS_LOOKUP[tuple(on)]
            else:
                digit = -1

            digits.append(digit)
            for seg in segments:
                cv2.rectangle(img, (seg[0][0]+x,seg[0][1]+y), (seg[1][0]+x,seg[1][1]+y), (0, 255, 0), 1)
            cv2.rectangle(img, (x,y),(x+w,y+h), (0, 255, 0), 1)
            cv2.putText(img, str(digit), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow("result", img)

        cv2.waitKey(0)


        result = 0
        digits.reverse()
        for i,dig in enumerate(digits):
            if dig == -1:
                return digits
            result += dig * (10**i)

        return result


image = cv2.imread("test.jpg")
SegmentProcessor.img_change(image)
SegmentProcessor.lcd_rect_setter()
result = SegmentProcessor.segment_getter()
print(result)