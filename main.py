from codecs import escape_encode
from multiprocessing.sharedctypes import Value
import cv2
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from imutils import contours as ctr
import imutils
from imutils.perspective import four_point_transform
from tkinter import filedialog
from tkinter import Tk
from os.path import exists
import pickle as pk
from multiprocessing import Process, Queue, Pipe
import matplotlib.pyplot as plt
plt.style.use('dark_background')



class SegmentProcessor:
    LOGGING_TIME_DIST = 1

    IMG_RESIZE = 8000
    IMG_SHEER = 0.0
    IMG_SHEER_MAT = np.array( [ [1,IMG_SHEER,0], 
                                [0,1,0]  ],dtype = np.float32 )
    DIGITS_HEIGHT_MAX = 0.8
    DIGITS_HEIGHT_MIN = 0.5

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
    
    digitMin = -1
    digitMax = -1
    MISSING_FAULT_LIMIT_E = 1.2
    MISSING_FAULT_LIMIT_C = 0.2

    rectpos = []

    @classmethod
    def setDigitRange(cls,min, max):
        cls.digitMin = min
        cls.digitMax = max
    
    @classmethod
    def lcd_rect_setter(cls,image):
        cls.rectpos = []
        img = imutils.resize(image, height=(1000))
        img = cv2.warpAffine(img, cls.IMG_SHEER_MAT, (0,0))
        img_copied = img.copy()
        #user select drawing range by clicking 4-pointed lcd rect.
        def on_mouse(event, x, y, flag, param):
            if event == cv2.EVENT_LBUTTONDOWN :
                cls.rectpos.append([[x,y]])
                cv2.circle(img, (x,y), 3, (0,255,0), -1)
                if len(cls.rectpos) >= 2:
                    cv2.line(img,cls.rectpos[-1][0],cls.rectpos[-2][0],(0,255,0),5)
                cv2.imshow("image", img)
                if len(cls.rectpos) == 4:
                    cv2.destroyAllWindows()
                    return
            if len(cls.rectpos) >= 1:
                imgcopied = img.copy()
                cv2.line(imgcopied,cls.rectpos[-1][0],(x,y),(0,255,0),2)
                cv2.imshow("image", imgcopied)


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
        oldpos = []
        def on_mouse(event, x, y, flag, param):
            if event == cv2.EVENT_LBUTTONDOWN :
                cls.rectpos.append([[xnew + x/1000*height_old,ynew + y/width_new*width_old]])
                oldpos.append((x,y))
                cv2.circle(img, (x,y), 3, (0,255,0), -1)
                if len(oldpos) >= 2:
                    cv2.line(img,oldpos[-1],oldpos[-2],(0,255,0),5)
                cv2.imshow("image", img)
                if len(cls.rectpos) >= 4:
                    cv2.destroyAllWindows()
                    return
            if len(oldpos) >= 1:
                imgcopied = img.copy()
                cv2.line(imgcopied,oldpos[-1],(x,y),(0,255,0),2)
                cv2.imshow("image", imgcopied)


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
    def segment_getter(cls, img, test):
        img = imutils.resize(img, height=(cls.IMG_RESIZE))

        height_now, w, c = img.shape
        rectpos_adjusted = np.array(cls.rectpos) * (height_now / 1000)
        img = four_point_transform(img, rectpos_adjusted.reshape(4,2))
        img = cv2.warpAffine(img, cls.IMG_SHEER_MAT, (0,0))
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


        img_blurred = cv2.GaussianBlur(gray, ksize=(0,0), sigmaX=cls.BLUR_BIAS)

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

        if test == True:
            cv2.imshow("check", img_blur_thresh)

        #contours making

        contours, _ = cv2.findContours(
            img_blur_thresh,
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_SIMPLE
        )

        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255,255,255))

        possible_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(temp_result, pt1=(x,y), pt2=(x+w, y+h), color=(255,255,255), thickness=2)
            area = w * h
            ratio = w / h
            
            if ratio < 1 and cls.DIGITS_HEIGHT_MIN * height < h < cls.DIGITS_HEIGHT_MAX * height:
                possible_contours.append(contour)

        try:
            digitCnts = ctr.sort_contours(possible_contours,
                method="left-to-right")[0]
        except ValueError:
            return ["x"]
        digits = []


        #check if there are missing digits. (check the "empty space" lik  2X3)
        boxes = []
        wid_list = []
        x_list = []
        for c in digitCnts:
            # extract the digit ROI
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append((x, y, w, h))
        for box in boxes:
            wid_list.append(box[2])
            x_list.append(box[0])
        x_delta = max(x_list) + wid_list[-1] - min(x_list)
        if x_delta / sum(wid_list) > cls.MISSING_FAULT_LIMIT_E:
            if test == True:
                digits.append("e")
            else:
                return ["e"]
            
        #check if there are missing digits. (check the "wrong coordinate" lik  23XX)
        if (width - max(x_list) - wid_list[x_list.index(max(x_list))]) / width > cls.MISSING_FAULT_LIMIT_C:
            if test == True:
                digits.append("c")
            else:
                return ["c"]

        # loop over each of the digits
        for c in digitCnts:
            # extract the digit ROI
            (x, y, w, h) = cv2.boundingRect(c)
            if h/w >= 2.0:
                digit = 1
                digits.append(digit)
                if test == True:
                    cv2.rectangle(img, (x,y),(x+w,y+h), (0, 255, 0), 1)
                    cv2.putText(img, str(digit), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                continue
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
                if area == 0.0:
                    return ["zeroth"]
                if total / float(area) > 0.5:
                    on[i]= 1
            # lookup the digit and draw it on the image

            if tuple(on) in cls.DIGITS_LOOKUP.keys():
                digit = cls.DIGITS_LOOKUP[tuple(on)]
            else:
                digit = -1

            digits.append(digit)
            if test == True:
                for seg in segments:
                    cv2.rectangle(img, (seg[0][0]+x,seg[0][1]+y), (seg[1][0]+x,seg[1][1]+y), (0, 255, 0), 1)
                cv2.rectangle(img, (x,y),(x+w,y+h), (0, 255, 0), 1)
                cv2.putText(img, str(digit), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        if test == True:
            cv2.imshow("result", img)



        result = 0
        digits.reverse()

        if  cls.digitMin > len(digits) or len(digits) > cls.digitMax :
            return digits
        for i,dig in enumerate(digits):
            if dig == -1 or type(dig) == type("str"):
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
    def preSampling(cls, amount):
        mod = cls.frameCount // amount
        vid = cv2.VideoCapture(cls.videoPath)
        sampled = []
        for i in range(cls.frameCount):
            if i % mod == 0:
                vid.set(cv2.CAP_PROP_POS_FRAMES, i)
                suc, img = vid.read()
                sampled.append(img)
        return sampled
        


#pre-process and find vid path
if __name__ == "__main__":
    cv2.destroyAllWindows()
    root = Tk()
    vidPath = filedialog.askopenfilename(initialdir="/", title="Select file",
                                            filetypes=(("m4v files", "*.m4v"),("mp4 files", "*.mp4"),
                                            ("all files", "*.*")))
    root.destroy()

    VideoFrames.vid_change(vidPath)
    vidnum = input("vid number  (int) ::")
    #VideoFrames.framestacking_depracated() => moved to main progress for lowering memory load.

    TEST_NUM = 10
    samples = VideoFrames.preSampling(TEST_NUM)
    SegmentProcessor.lcd_rect_setter(samples[TEST_NUM // 2])

    digitMin = int(input("please put the minimum digits of your device : "))
    digitMax = int(input("please put the Maximum digits of your device : "))
    SegmentProcessor.setDigitRange(digitMin, digitMax)


    if exists(VideoFrames.videoPath+".sslg.spec"):
        with open(VideoFrames.videoPath+vidnum+".sslg.spec", "rb") as f:
            settinglist = pk.load(f)
            SegmentProcessor.BLUR_BIAS = settinglist[0]
            SegmentProcessor.IMG_RESIZE = settinglist[1]
            SegmentProcessor.IMG_SHEER = settinglist[2]
            SegmentProcessor.DIGITS_HEIGHT_MAX = settinglist[3]
            SegmentProcessor.DIGITS_HEIGHT_MIN = settinglist[4]
            SegmentProcessor.MISSING_FAULT_LIMIT_C = settinglist[5]
            SegmentProcessor.MISSING_FAULT_LIMIT_C = settinglist[6]
            
    else:
        while True:
            for i in range(TEST_NUM):
                result = SegmentProcessor.segment_getter(samples[i], True)
                if type(result) == type([]):
                    sum_text = ""
                    if len(result) == 1:
                        sum_text = str(result[0])
                    result.reverse()  
                    for res in result:
                        if res == -1:
                            sum_text += "X"
                            continue
                        sum_text += str(res)
                    print(sum_text)
                else:
                    print(result)
                cv2.waitKey(100)
            acceptable = ""
            while True:
                acceptable = input("check again : any, start process : 1, adjust blur : 2, adjust size : 3, adjust sheering : 4, digit's height min : 5,\n adjust digit's height MAX : 6, adjust FAULT_LIMIT (e, 2X45) : 7, adjust FAULT_LIMIT (c, 24XX) : 8 \n      :: ")
                if acceptable == "1":
                    break
                elif acceptable == "2":
                    cv2.destroyAllWindows()
                    print("current blur : " , SegmentProcessor.BLUR_BIAS)
                    val = input(" change to  : ")
                    val = int(val)
                    SegmentProcessor.BLUR_BIAS = val
                elif acceptable == "3":
                    cv2.destroyAllWindows()
                    print("current size factor : " , SegmentProcessor.IMG_RESIZE)
                    val = input("re-factor to (integer) : ")
                    SegmentProcessor.IMG_RESIZE = int(val)
                elif acceptable == "4":
                    cv2.destroyAllWindows()
                    print("current sheer factor : " , SegmentProcessor.IMG_SHEER)
                    val = input("re-factor to (float) : ")
                    SegmentProcessor.IMG_SHEER = float(val)
                    SegmentProcessor.IMG_SHEER_MAT = np.array( [ [1,SegmentProcessor.IMG_SHEER,0], 
                                                                [0,1,0]  ],dtype = np.float32 )
                elif acceptable == "5":
                    cv2.destroyAllWindows()
                    print("current digit's height min : " , SegmentProcessor.DIGITS_HEIGHT_MIN)
                    val = input("re-factor to (float) : ")
                    SegmentProcessor.DIGITS_HEIGHT_MIN = float(val)
                elif acceptable == "6":
                    cv2.destroyAllWindows()
                    print("current digit's height MAX : " , SegmentProcessor.DIGITS_HEIGHT_MAX)
                    val = input("re-factor to (float) : ")
                    SegmentProcessor.DIGITS_HEIGHT_MAX = float(val)
                elif acceptable == "7":
                    cv2.destroyAllWindows()
                    print("current e_threshHold (1~0 ratio, digit's distance is bigger than ratio, then faulted.) : " , SegmentProcessor.MISSING_FAULT_LIMIT_E)
                    val = input("re-factor to (float) : ")
                    SegmentProcessor.MISSING_FAULT_LIMIT_E = float(val)
                elif acceptable == "8":
                    cv2.destroyAllWindows()
                    print("current c_threshHold (1~0 ratio, left blank is bigger than ratio, then faulted.) : " , SegmentProcessor.MISSING_FAULT_LIMIT_C)
                    val = input("re-factor to (float) : ")
                    SegmentProcessor.MISSING_FAULT_LIMIT_C = float(val)
                else:
                    break
            if acceptable == "1":
                break

        print("done!", SegmentProcessor.BLUR_BIAS, SegmentProcessor.IMG_RESIZE, SegmentProcessor.IMG_SHEER,SegmentProcessor.DIGITS_HEIGHT_MAX,SegmentProcessor.DIGITS_HEIGHT_MIN,SegmentProcessor.MISSING_FAULT_LIMIT_C, SegmentProcessor.MISSING_FAULT_LIMIT_E)

        with open(VideoFrames.videoPath+vidnum+".sslg.spec", "wb") as f:
            setting = [SegmentProcessor.BLUR_BIAS, SegmentProcessor.IMG_RESIZE, SegmentProcessor.IMG_SHEER,SegmentProcessor.DIGITS_HEIGHT_MAX,SegmentProcessor.DIGITS_HEIGHT_MIN,SegmentProcessor.MISSING_FAULT_LIMIT_C, SegmentProcessor.MISSING_FAULT_LIMIT_E]
            pk.dump(setting, f)
        
        with open(VideoFrames.videoPath+vidnum+".sslg.spec.txt", "w") as f:
            setting = [SegmentProcessor.BLUR_BIAS, SegmentProcessor.IMG_RESIZE, SegmentProcessor.IMG_SHEER,SegmentProcessor.DIGITS_HEIGHT_MAX,SegmentProcessor.DIGITS_HEIGHT_MIN,SegmentProcessor.MISSING_FAULT_LIMIT_C, SegmentProcessor.MISSING_FAULT_LIMIT_E]
            f.write(str(setting))



#main progress.

def mainprogress(id,vidPath, start, end,step,fps,dmin, dmax,
                blurbias, imgresize, imgsheer, heightmax, heightmin, 
                faulte, faultc, result, pipeline):
    rectpos = pipeline.recv()
    SegmentProcessor.rectpos = rectpos
    SegmentProcessor.digitMin = dmin
    SegmentProcessor.digitMax = dmax
    SegmentProcessor.BLUR_BIAS = blurbias
    SegmentProcessor.IMG_RESIZE = imgresize
    SegmentProcessor.IMG_SHEER = imgsheer
    SegmentProcessor.IMG_SHEER_MAT = np.array( [ [1,SegmentProcessor.IMG_SHEER,0], 
                                                    [0,1,0]  ],dtype = np.float32 )
    SegmentProcessor.DIGITS_HEIGHT_MAX = heightmax
    SegmentProcessor.DIGITS_HEIGHT_MIN = heightmin

    SegmentProcessor.MISSING_FAULT_LIMIT_E = faulte
    SegmentProcessor.MISSING_FAULT_LIMIT_C = faultc
    vid = cv2.VideoCapture(vidPath)
    length = ((end - start) // step) + 1
    for i in range(length):
        if i < length - 1:
            i = i * step + start
        else:
            i = end - 1
        vid.set(cv2.CAP_PROP_POS_FRAMES, i)
        suc,img = vid.read()
        if not suc:
            raise Exception("index overflow." + str(i))
        number = SegmentProcessor.segment_getter(img, False)
        if type(number) == type([]):
            continue
        else:
            time = i / fps
            number = (number, time)    
        result.put(number)  
    print(id, "done!")   
        
if __name__ == "__main__":
    print("data logging in progress...")
    cv2.destroyAllWindows()
    logged = []
    if VideoFrames.frameCount >1000:
        GROUPED = 1000
        queues = []
        threads = []
        divided = VideoFrames.frameCount // GROUPED  
        divided_time = GROUPED / VideoFrames.fps
        for i in range(divided + 1):
            start = i * GROUPED
            end = 0
            if i == divided:
                end = VideoFrames.frameCount
            else:
                end = (i+1) * GROUPED
            queues.append(Queue())
            segmentprocessor_pipe_parent, segmentprocessor_pipe_child = Pipe()
            segmentprocessor_pipe_parent.send(deepcopy(SegmentProcessor.rectpos))
            threads.append(Process(target=mainprogress, args=(i, VideoFrames.videoPath ,start, end,
                            int(VideoFrames.fps*SegmentProcessor.LOGGING_TIME_DIST), VideoFrames.fps,SegmentProcessor.digitMin, SegmentProcessor.digitMax
                            ,SegmentProcessor.BLUR_BIAS,SegmentProcessor.IMG_RESIZE,SegmentProcessor.IMG_SHEER,SegmentProcessor.DIGITS_HEIGHT_MAX,SegmentProcessor.DIGITS_HEIGHT_MIN, 
                            SegmentProcessor.MISSING_FAULT_LIMIT_E,SegmentProcessor.MISSING_FAULT_LIMIT_C, queues[i], segmentprocessor_pipe_child)))
        for i in range(divided + 1):
            threads[i].start()
        for i in range(divided + 1):
            threads[i].join()
        for i in range(divided + 1):
            queues[i].put("STOP")
        for i in range(divided + 1):
            while True:
                got = queues[i].get()
                if got == "STOP":
                    break
                else:
                    logged.append(got)
        print("got : ", len(logged), "would be :", VideoFrames.frameCount//(VideoFrames.fps*SegmentProcessor.LOGGING_TIME_DIST))
    else:
        samples = VideoFrames.preSampling(VideoFrames.frameCount//(VideoFrames.fps*SegmentProcessor.LOGGING_TIME_DIST))
        for i in range(len(samples)):
            img = samples[i]
            result = SegmentProcessor.segment_getter(img, False)
            if type(result) == type([]):
                continue
            else:
                time = i * SegmentProcessor.LOGGING_TIME_DIST
                logged.append((result, time))





    #afterprocess. making csv.
    logged.sort(key=lambda x:x[1])
    with open(VideoFrames.videoPath +vidnum+ "_result.csv", "w") as f:
        print(logged[0])
        timestr = str(logged[0][1])
        for i in range(1,len(logged)):
            timestr += ","+str(logged[i][1])
        f.write(timestr + "\n")

        datastr = str(logged[0][0])
        for i in range(1,len(logged)):
            datastr += ","+str(logged[i][0])
        f.write(datastr +"\n")

    print(logged)
    logged = list(zip(*logged))
    plt.scatter(logged[1],logged[0])
    plt.show()



    