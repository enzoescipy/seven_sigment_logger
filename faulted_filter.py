from multiprocessing.sharedctypes import Value
from scipy.signal import lfilter
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter import filedialog
from copy import deepcopy

SLOPE_THRESH = 100
vidPath = filedialog.askopenfilename(initialdir="/", title="Select file",
                                        filetypes=(("csv files", "*.csv"),
                                        ("all files", "*.*")))
#vidPath = "testvid/3/20220712_095921.mp4_result.csv"

while True:
    logged = []

    root = Tk()

    root.destroy()

    fig, ax = plt.subplots()

    with open(vidPath, "r") as f:
        times = f.readline()[:-1].split(",")
        datas = f.readline()[:-1].split(",")
        while True:
            try:
                times.remove("")
                datas.remove("")
            except ValueError:
                break
        times = list(map(float,times))
        datas = list(map(float,datas))
        logged = list(zip(times, datas))
    ax.plot(times,datas)
    i=0
    while True:
        if i == 0:
            i+= 1
            continue
        elif i >= len(logged) - 1:
            break
        log_bef = logged[i-1]
        log = logged[i]
        log_aft = logged[i+1]
        if (log_bef[0] - log[0]) == 0.0 or (log[0] - log_aft[0]) == 0.0:
            logged.pop(i)
            i-=1
            continue
        slope_now = (log_bef[1] - log[1]) / (log_bef[0] - log[0])
        slope_aft = (log[1] - log_aft[1]) / (log[0] - log_aft[0])
        print(np.abs(slope_aft - slope_now))

        if np.abs(slope_aft - slope_now) > SLOPE_THRESH:
            logged.pop(i+1)
            i-=1
            continue
        i+=1

    times, datas = list(zip(*logged))


    ax.plot(times,datas)
    plt.show()

    ans = input("save result? y : yes, n : no    ::")
    if ans == "y":
        with open(vidPath, "w") as f:
            timestr = str(logged[0][0])
            for i in range(1,len(logged)):
                timestr += ","+str(logged[i][0])
            f.write(timestr + "\n")

            datastr = str(logged[0][1])
            for i in range(1,len(logged)):
                datastr += ","+str(logged[i][1])
            f.write(datastr +"\n")
        break
    else:
        print("current threshHold value : ", SLOPE_THRESH)
        ans = input("renew value(int) : ")
        SLOPE_THRESH = int(ans)
