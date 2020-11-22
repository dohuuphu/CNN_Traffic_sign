import json
import os
import glob
import csv
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data_path = pd.read_csv('data_train.csv')
lenght = len
a=len(data_path.iloc[:, 1])
print(a)
lt=[]
for i in range(a):
	a=data_path.iloc[i, 1]
	a=a
	lt.append(a)
plt.hist(lt)
plt.show()

