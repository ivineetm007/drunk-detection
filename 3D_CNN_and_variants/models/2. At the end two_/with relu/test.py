import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from model import *

import csv 

from torchvision import transforms, utils
import pandas as pd

import cv2

from torch.utils.data import Dataset


class VideoDataset(Dataset):
	"""Dataset Class for Loading Drunk, Sober Video"""

	def __init__(self, mode, folder = "./data/videos/", file = "video_files.csv"):
		"""
		Args:
			folder: where videos are there
			file: the csv file
			mode: mode of the dataset i.e., train, val, test
		"""

		self.mode = mode
		self.files = [] #the files
		self.cats = [] #the folders

		train = pd.read_csv("./data/" + file, header = None)
		for index in range(len(train[0])):
			if (train[0][index] == self.mode):
				file = folder + train[2][index]
				category = 1.0 if train[1][index] == "Drunk" else 0.0

				self.files.append(file)
				self.cats.append(category)

		self.length = len(self.files)

		self.transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
						transforms.Normalize([0.29075419,0.34523574,0.47825202], [0.1413886,0.13252793,0.15167585])])

	def __len__(self):
		return self.length

	def readVideo(self, videoFile):

		cap = cv2.VideoCapture(videoFile)

		frames = torch.FloatTensor(3, 240, 224, 224)  #hardcoding 240 frames, size is 224 X 224
	
		for f in range(240):

			ret, frame = cap.read()
	
			if ret:
				frame = self.transform(frame)
				frames[:, f, :, :] = frame

			else:
				break
	
		return frames

	def __getitem__(self, idx):

		file = self.readVideo(self.files[idx])
		category = self.cats[idx]

		return file, self.files[idx], category

batch_size = 4

#data loading 
test_data = VideoDataset(mode = "test") 
test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 8)
test_data_len = len(test_data)


#model initialization and multi GPU
device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")

#resnet not pre-trained one
model = Model()

best_state = torch.load("best_3Dconv.pt") #load the best state
model.load_state_dict(best_state['state_dict']) #load the weights

model = model.to(device)

result = ""

model.eval()
acc = 0.0

for i, (inputs,files,labels) in enumerate(test_loader):

	inputs = inputs.to(device) #change to device
	labels = labels.to(device)

	with torch.no_grad(): 
		predictions = model(inputs)

	print (predictions)

	for j in range(len(files)):
		result += str(files[j])
		result += ","
		result = result +   str(float(1 - predictions[j][0].item())) #if more than 0.5 then drunk
		result += ","
		result = result +   str(float(predictions[j][0].item())) #if more than 0.5 then drunk
		result += "\n"

	predictions = predictions >= 0.5  # give abnormal label
		
	acc += (predictions.squeeze().float() == labels.float()).sum()

print (acc, float(test_data_len) )
print ("Accuracy: ")
print (acc.float() / float(test_data_len))

f = open ("result.csv", 'w')
f.write(result)
f.close()
