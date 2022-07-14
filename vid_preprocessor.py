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

