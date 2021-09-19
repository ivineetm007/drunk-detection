import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class Model(nn.Module):

	def __init__(self):
		super(Model, self).__init__()

		self.device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")

		self.maxpool = nn.MaxPool3d(kernel_size = 2, stride = 2)

		self.conv1 = nn.Conv3d(3, 64, kernel_size = 3, stride = (2, 2, 2), padding = (1, 1, 1))
		self.bn1 = nn.BatchNorm3d(64)

		self.conv2 = nn.Conv3d(64, 64, kernel_size = 3, stride = (2, 2, 2), padding = (1, 1, 1))
		self.bn2 = nn.BatchNorm3d(64)


		'''Third one'''
		self.conv3_hw = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
		self.bn3_hw = nn.BatchNorm2d(128)

		self.conv3_lh = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
		self.bn3_lh = nn.BatchNorm2d(128)

		self.conv3_lw = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
		self.bn3_lw = nn.BatchNorm2d(128)


		'''Fourth One'''
		self.conv4_hw = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
		self.bn4_hw = nn.BatchNorm2d(128)

		self.conv4_lh = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
		self.bn4_lh = nn.BatchNorm2d(128)

		self.conv4_lw = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
		self.bn4_lw = nn.BatchNorm2d(128)

		'''Classfication Layer'''
		self.fc = nn.Linear(128 * 15 * 14 * 14, 1)
		self.probability = nn.Sigmoid()


	def forward(self, x):

		out = self.bn1(F.relu(self.conv1(x)))

		out = self.bn2(F.relu(self.conv2(out)))

		'''Third Conv'''
		out1 = torch.zeros([x.shape[0], 128, 60, 56, 56]).to(self.device)

		for i in range(out1.shape[2]): #hw
			out1[:,:,i,:,:] += self.bn3_hw(self.conv3_hw(out[:,:,i,:,:]))

		for i in range(out1.shape[3]): #Lw
			out1[:,:,:,i,:] += self.bn3_lw(self.conv3_lw(out[:,:,:,i,:]))

		for i in range(out1.shape[4]): #Lh
			out1[:,:,:,:,i] += self.bn3_lh(self.conv3_lh(out[:,:,:,:,i]))

		out = self.maxpool(F.relu(out1, True))


		'''Fourth Conv'''

		out1 = torch.zeros([x.shape[0], 128, 30, 28, 28]).to(self.device)

		for i in range(out1.shape[2]): #hw
			out1[:,:,i,:,:] += self.bn4_hw(self.conv4_hw(out[:,:,i,:,:]))

		for i in range(out1.shape[3]): #Lw
			out1[:,:,:,i,:] += self.bn4_lw(self.conv4_lw(out[:,:,:,i,:]))

		for i in range(out1.shape[4]): #Lh
			out1[:,:,:,:,i] += self.bn4_lh(self.conv4_lh(out[:,:,:,:,i]))

		out = self.maxpool(F.relu(out1, True))

		'''Classfn'''
		out = self.fc(out.view( out.shape[0], -1))

		out = self.probability(out)

		return out
