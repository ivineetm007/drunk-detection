import cv2
import pandas as pd

import numpy as np

import os

mean = np.zeros(3)
std = np.zeros(3)
count = 0

sums = 0
sum_squares = 0

train = pd.read_csv("video_files.csv", header = None)

train_files = []

for index in range(len(train[0])):
	if (train[0][index] == "train"):
		file = "./videos/" + train[1][index] + "/" + train[2][index]
		train_files.append(file)


for file in train_files:

	cap = cv2.VideoCapture(file)
	
	for f in range(300):

		ret, frame = cap.read()
		
		if ret:

			data = np.asarray( frame, dtype='float' ) / 255.0
			data = data.reshape(-1, 3).T 

			sample_mean = np.mean(data, axis = 1)

			sums += sample_mean
			sum_squares += (sample_mean ** 2)

			count += 1            

		else:
			print("Skipped!" + str(f))
			break
		


	print (count)

sums /= float(count)

print ("mean:")
print (sums)

std = -((sums ** 2) - (sum_squares / float(count)))
std = (std ** 0.5)

print ("std:")
print (std)

mean:
[0.29075419 0.34523574 0.47825202]

std:
[0.1413886  0.13252793 0.15167585]

