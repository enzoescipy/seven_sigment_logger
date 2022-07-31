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

logged = []


fig, ax = plt.subplots()

with open(vidPath, "r") as f:
    times1 = list(map(float,f.readline()[:-1].split(",")))
    datas1 = list(map(float,f.readline()[:-1].split(",")))
    logged1 = list(zip(times1, datas1))
ax.plot(times1,datas1)
plt.show()

vid1name = vidPath.split(".")[-2]

vidPath = filedialog.askopenfilename(initialdir="/", title="Select file",
                                        filetypes=(("csv files", "*.csv"),
                                        ("all files", "*.*")))
#vidPath = "testvid/3/20220712_095921.mp4_result.csv"

logged = []


fig, ax = plt.subplots()

with open(vidPath, "r") as f:
    times2 = list(map(float,f.readline()[:-1].split(",")))
    datas2 = list(map(float,f.readline()[:-1].split(",")))
    logged2 = list(zip(times2, datas2))
ax.plot(times2,datas2)
plt.show()

logged = logged1 + logged2
logged.sort(key = lambda log:log[0])

unpacked = list(zip(*logged))
times = unpacked[0]
datas = unpacked[1]

fig, ax = plt.subplots()

ax.plot(times,datas)
plt.show()
vidPath =  filedialog.asksaveasfilename(
                defaultextension='.csv', filetypes=[("csv files", '*.csv')],
                initialdir="/",
                title="Choose filename")
print("save result? y : yes, n : no")
if input(":") == "y":
    with open(vidPath, "w") as f:
        timestr = str(logged[0][0])
        for i in range(1,len(logged)):
            timestr += ","+str(logged[i][0])
        f.write(timestr + "\n")

        datastr = str(logged[0][1])
        for i in range(1,len(logged)):
            datastr += ","+str(logged[i][1])
        f.write(datastr +"\n")