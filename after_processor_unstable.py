import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative
import scipy.signal as signal
from scipy.interpolate import splrep, splev, splder
from tkinter import Tk
from tkinter import filedialog

TIME_DISTANCE_MINIMUM = 0.5
SAVGOL_PARA = 5

exprs = ["10 ","40 ","70 "]
iters = ["1 ","2 ","3 "]
parts = ["server","total"]

dataDict = {}
vidPath = filedialog.askopenfilename(initialdir="/", title="Select file",
                                        filetypes=(("csv files", "*.csv"),
                                        ("all files", "*.*")))
directory = vidPath
with open(directory, "r") as f:
    times = f.readline()[:-1].split(",")
    datas = f.readline()[:-1].split(",")

    times = list(map(float,times))
    datas = list(map(float,datas))
    
    # skipping too close in time points.
    i = 0
    while True:
        if i >= len(times) - 1:
            break
        time = times[i]
        timenext = times[i+1]
        if timenext - time < TIME_DISTANCE_MINIMUM:
            times.pop(i+1)                           
            datas.pop(i+1)
            i -= 1 
        i += 1
        


times, datas = (times, datas)
times = np.array(times)
datas = np.array(datas)
timedata_funcorig = splrep(times, datas,k=2)
new_time = np.linspace(times[0],times[-1], 200)
datas = splev(new_time, timedata_funcorig)
datas = signal.savgol_filter(datas, SAVGOL_PARA, 2)
plt.plot(new_time, datas)
plt.show()

        
