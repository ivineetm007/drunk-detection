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

		self.conv3_l = nn.Conv1d(128, 128, kernel_size = 3, padding = 1)
		self.bn3_l = nn.BatchNorm1d(128)


		'''Fourth One'''
		self.conv4_hw = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
		self.bn4_hw = nn.BatchNorm2d(128)

		self.conv4_l = nn.Conv1d(128, 128, kernel_size = 3, padding = 1)
		self.bn4_l = nn.BatchNorm1d(128)

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

		out1 = out1.permute(0,3,4,1,2)
		out1 = out1.contiguous().view(-1, 128, 60)

		out1 = self.bn3_l ( self.conv3_l(out1) )

		out1 = out1.contiguous().view(x.shape[0], 128, 60, 56, 56)

		out = self.maxpool(F.relu(out1, True))

		'''Fourth Conv'''

		out1 = torch.zeros([x.shape[0], 128, 30, 28, 28]).to(self.device)

		for i in range(out1.shape[2]): #hw
			out1[:,:,i,:,:] += self.bn4_hw(self.conv4_hw(out[:,:,i,:,:]))

		out1 = out1.permute(0,3,4,1,2)
		out1 = out1.contiguous().view(-1, 128, 30)

		out1 = self.bn4_l ( self.conv4_l(out1) )

		out1 = out1.contiguous().view(x.shape[0], 128, 30, 28, 28)

		out = self.maxpool(F.relu(out1, True))

		'''Classfn'''
		out = self.fc(out.view( out.shape[0], -1))

		out = self.probability(out)

		return out
