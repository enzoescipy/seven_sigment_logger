from scipy.signal import lfilter
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter import filedialog
from copy import deepcopy

logged = []

'''
root = Tk()
vidPath = filedialog.askopenfilename(initialdir="/", title="Select file",
                                        filetypes=(("csv files", "*.csv"),
                                        ("all files", "*.*")))
root.destroy()
'''
vidPath = "D:/aa_Projects_Folder/seven_segment_logger/examples/dot_removed.mp4_result.csv"

with open(vidPath, "r") as f:
    times = list(map(float,f.readline()[:-1].split(",")))
    datas = list(map(float,f.readline()[:-1].split(",")))
    logged = list(zip(times, datas))



logged = list(zip(*logged))

fig, ax = plt.subplots()

ax.scatter(logged[0],logged[1])
plt.show()
