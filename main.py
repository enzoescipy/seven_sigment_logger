from codecs import escape_encode
import cv2
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from imutils import contours as ctr
import imutils
from imutils.perspective import four_point_transform
from tkinter import filedialog
from tkinter import Tk
plt.style.use('dark_background')


class SegmentProcessor:
    LOGGING_RATE = 0.1

    IMG_RESIZE = 8000

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
    GRAYOUT_METHOD = (1,0,0) #b, g, r order.
    ROTATION = None
    REVERSE_COLOR = True
    THERSH_HOLD_GLOBAL = False

    rectpos = []
    @classmethod
    def img_change(cls, image) -> None:
        cls.image = image


    
    @classmethod
    def lcd_rect_setter(cls):
        cls.rectpos = []
        img = imutils.resize(cls.image, height=(1000))
        img_copied = img.copy()
        #user select drawing range by clicking 4-pointed lcd rect.
        def on_mouse(event, x, y, flag, param):
            if event == cv2.EVENT_LBUTTONDOWN :
                cls.rectpos.append([[x,y]])
                cv2.circle(img, (x,y), 3, (0,255,0), -1)
                cv2.imshow("image", img)
                if len(cls.rectpos) == 4:
                    cv2.destroyAllWindows()
                    return

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_mouse, img)

        #more presize croping.
        cv2.imshow("image", img)
        cv2.waitKey(0)
        xs = []
        ys = []
        for point in cls.rectpos:
            point = point[0]
            xs.append(point[0])
            ys.append(point[1])
        xs.sort()
        ys.sort()
        ynew = int(ys[0]*0.9)
        hnew = int(ys[-1]*1.1)
        xnew = int(xs[0]*0.9)
        wnew = int(xs[-1]*1.1)

        img_original = img_copied.copy()
        img = img_copied[ynew:hnew, xnew:wnew]
        height_old, width_old, c = img.shape
        img = imutils.resize(img, height=(1000))
        h,width_new, c = img.shape
        #user select drawing range by clicking 4-pointed lcd rect.
        cls.rectpos = []
        def on_mouse(event, x, y, flag, param):
            if event == cv2.EVENT_LBUTTONDOWN :
                cls.rectpos.append([[xnew + x/1000*height_old,ynew + y/width_new*width_old]])
                cv2.circle(img, (x,y), 3, (0,255,0), -1)
                cv2.imshow("image", img)
                if len(cls.rectpos) >= 4:
                    cv2.destroyAllWindows()
                    return

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_mouse, img)
        cv2.imshow("image", img)
        cv2.waitKey(0)

        print("WAIT!! is the image has to be rotated? 0: NO. keep going. -:counter-clockwise +:clockwise 1:flip")
        img_original = four_point_transform(img_original, np.array(cls.rectpos).reshape(4,2))
        cv2.imshow("image", img_original)
        cv2.waitKey(0)
        rotating = input("::  ")

        if rotating == "-":
            cls.ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE
            print("image will be ROTATE_90_COUNTERCLOCKWISE")
        elif rotating == "+":
            cls.ROTATION = cv2.ROTATE_90_CLOCKWISE
            print("image will be ROTATE_90_CLOCKWISE")
        elif rotating == "1":
            cls.ROTATION = cv2.ROTATE_180
            print("image will be ROTATE_180")
             

    @classmethod
    def segment_getter(cls):
        img = imutils.resize(cls.image, height=(cls.IMG_RESIZE))
        height_now, w, c = img.shape
        rectpos_adjusted = np.array(cls.rectpos) * (height_now / 1000)
        img = four_point_transform(img, rectpos_adjusted.reshape(4,2))
        if cls.ROTATION != None:
            img = cv2.rotate(img, cls.ROTATION)
        #image preprocessing
        height, width, channel = img.shape
        if cls.GRAYOUT_METHOD != (1,1,1):
            for i in range(3):
                switch = cls.GRAYOUT_METHOD[i]
                if switch == 0:
                    img[:,:,0] = np.zeros([img.shape[0],img.shape[1]])

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
                C=5
            )


        if cls.REVERSE_COLOR:
            img_blur_thresh = 255 - img_blur_thresh

        cv2.imshow("check", img_blur_thresh)

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
            
            if d['h'] > height*0.7 and  ratio < 1:
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
            if h/w >= 4:
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



        result = 0
        digits.reverse()
        for i,dig in enumerate(digits):
            if dig == -1:
                return digits
            result += dig * (10**i)

        return result

class VideoFrames:
    video = None
    frames = []
    @classmethod
    def vid_change(cls, videoPath):
        cls.video = cv2.VideoCapture(videoPath)
        cls.videoPath = videoPath
        cls.fps = cls.video.get(cv2.CAP_PROP_FPS)
        cls.frameCount = int(cls.video.get(cv2.CAP_PROP_FRAME_COUNT))

    @classmethod
    def framestacking(cls):
        count = 0
        while True:
            succes, image = cls.video.read()
            if not succes:
                break
            cls.frames.append(cv2.rotate(image, cv2.ROTATE_180))
            count += 1
            if count % 99 == 0:
                print(int(100 * count / cls.frameCount))

    @classmethod        
    def sampling(cls, amount):
        mod = cls.frameCount // amount
        sampled = []
        for i in range(cls.frameCount):
            if i % mod == 0:
                sampled.append(cls.frames[i])
        return sampled
        


#pre-process and find vid path
root = Tk()
vidPath = filedialog.askopenfilename(initialdir="/", title="Select file",
                                          filetypes=(("mp4 files", "*.mp4"),
                                          ("all files", "*.*")))
root.destroy()

VideoFrames.vid_change(vidPath)
VideoFrames.framestacking()

TEST_NUM = 6
samples = VideoFrames.sampling(TEST_NUM)
SegmentProcessor.img_change(samples[TEST_NUM // 2])
SegmentProcessor.lcd_rect_setter()

while True:
    for i in range(TEST_NUM):
        SegmentProcessor.img_change(samples[i])
        result = SegmentProcessor.segment_getter()
        if type(result) == type([]):
            sum = ""
            result.reverse()
            for res in result:
                if res == -1:
                    sum += "X"
                    continue
                sum += str(res)
            print(sum)
        else:
            print(result)
        cv2.waitKey(3000)
    
    acceptable = input("keep going : 1, adjust blur : 2, adjust size : 3       :: ")
    if acceptable == "2":
        cv2.destroyAllWindows()
        print("current blur : " , SegmentProcessor.BLUR_BIAS)
        bias = input("- : minus 1, + : plus 1, 0 : ignore")
        if bias == "-" : 
            bias = -1
        elif bias == "+" : 
            bias = 1
        else: 
            bias = 0
        SegmentProcessor.BLUR_BIAS +=  bias
    elif acceptable == "3":
        cv2.destroyAllWindows()
        print("current size factor : " , SegmentProcessor.IMG_RESIZE)
        bias = input("re-factor to (integer) : ")
        SegmentProcessor.IMG_RESIZE = int(bias)
    else:
        break

print("done!", SegmentProcessor.BLUR_BIAS, SegmentProcessor.IMG_RESIZE)
with open(VideoFrames.videoPath+"_spec.txt", "w") as f:
    f.write("done!\n")
    f.write(str(SegmentProcessor.BLUR_BIAS)+"\n")
    f.write(str(SegmentProcessor.IMG_RESIZE)+"\n")




#main progress.
samples = VideoFrames.sampling(VideoFrames.frameCount//(VideoFrames.fps*SegmentProcessor.LOGGING_RATE))
logged = []
for i in range(len(samples)):
    SegmentProcessor.img_change(samples[i])
    result = SegmentProcessor.segment_getter()
    if type(result) == type([]):
        continue
    else:
        time = i * SegmentProcessor.LOGGING_RATE
        logged.append((result, time))

with open(VideoFrames.videoPath + "_result.csv", "w") as f:
    timestr = str(logged[0][1])
    for i in range(1,len(logged)):
        timestr += ","+str(logged[i][1])
    f.write(timestr + "\n")



    datastr = str(logged[0][0])
    for i in range(1,len(logged)):
        datastr += ","+str(logged[i][0])
    f.write(datastr +"\n")