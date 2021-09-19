import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class Model(nn.Module):

	def __init__(self):
		super(Model, self).__init__()

		self.device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")

		self.maxpool = nn.MaxPool3d(kernel_size = 2, stride = 2)

		'''First one'''
		#self.conv1_hw = nn.Conv2d(3, 64, kernel_size = 3, padding = 1)
		#self.bn1_hw = nn.BatchNorm2d(64)

		#self.conv1_lh = nn.Conv2d(3, 64, kernel_size = 3, padding = 1)
		#self.bn1_lh = nn.BatchNorm2d(64)

		#self.conv1_lw = nn.Conv2d(3, 64, kernel_size = 3, padding = 1)
		#self.bn1_lw = nn.BatchNorm2d(64)


		'''Second one'''
		#self.conv2_hw = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
		#self.bn2_hw = nn.BatchNorm2d(64)

		#self.conv2_lh = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
		#self.bn2_lh = nn.BatchNorm2d(64)

		#self.conv2_lw = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
		#self.bn2_lw = nn.BatchNorm2d(64)

		self.conv1 = nn.Conv3d(3, 64, kernel_size = 3, stride = (2, 2, 2), padding = (1, 1, 1))
		self.bn1 = nn.BatchNorm3d(64)

		#self.conv2 = nn.Conv3d(64, 64, kernel_size = 3, stride = (2, 2, 2), padding = (1, 1, 1))
		#self.bn2 = nn.BatchNorm3d(64)
		
		'''Second one'''
		self.conv2_hw = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
		self.bn2_hw = nn.BatchNorm2d(64)

		self.conv2_l = nn.Conv1d(64, 64, kernel_size = 3, padding = 1)
		self.bn2_l = nn.BatchNorm1d(64)

		'''Third one'''
		self.conv3_hw = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
		self.bn3_hw = nn.BatchNorm2d(128)

		self.conv3_l = nn.Conv1d(128, 128, kernel_size = 3, padding = 1)
		self.bn3_l = nn.BatchNorm1d(128)

		#self.conv3_lh = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
		#self.bn3_lh = nn.BatchNorm2d(128)

		#self.conv3_lw = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
		#self.bn3_lw = nn.BatchNorm2d(128)


		'''Fourth One'''
		self.conv4_hw = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
		self.bn4_hw = nn.BatchNorm2d(128)

		self.conv4_l = nn.Conv1d(128, 128, kernel_size = 3, padding = 1)
		self.bn4_l = nn.BatchNorm1d(128)


		#self.conv4_lh = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
		#self.bn4_lh = nn.BatchNorm2d(128)

		#self.conv4_lw = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
		#self.bn4_lw = nn.BatchNorm2d(128)

		'''Classfication Layer'''
		self.fc = nn.Linear(128 * 15 * 14 * 14, 1)
		self.probability = nn.Sigmoid()


	def forward(self, x):

		'''First Conv'''

		'''out = torch.zeros([1, 64, 240, 224, 224]).to(self.device)

		for i in range(out.shape[2]): #hw
			out[:,:,i,:,:] += (self.conv1_hw(x[:,:,i,:,:]))

		for i in range(out.shape[3]): #Lw
			out[:,:,:,i,:] += (self.conv1_lw(x[:,:,:,i,:]))

		for i in range(out.shape[4]): #Lh
			out[:,:,:,:,i] += (self.conv1_lh(x[:,:,:,:,i]))'''

		'''out = self.maxpool(F.relu(out, True))

		out1 = torch.zeros([1, 64, 120, 112, 112]).to(self.device)

		for i in range(out1.shape[2]): #hw
			out1[:,:,i,:,:] += (self.conv2_hw(out[:,:,i,:,:]))

		for i in range(out1.shape[3]): #Lw
			out1[:,:,:,i,:] += (self.conv2_lw(out[:,:,:,i,:]))

		for i in range(out1.shape[4]): #Lh
			out1[:,:,:,:,i] += (self.conv2_lh(out[:,:,:,:,i]))'''

		#out = self.maxpool(F.relu(out1, True))

		out = self.bn1(F.relu(self.conv1(x)))
		
		'''Second Conv'''
		out1 = torch.zeros([x.shape[0], 64, 120, 112, 112]).to(self.device)

		for i in range(out1.shape[2]): #hw
			out1[:,:,i,:,:] += F.relu (self.bn2_hw(self.conv2_hw(out[:,:,i,:,:])), True)

		'''for i in range(out1.shape[3]): #Lw
			out1[:,:,:,i,:] += self.bn3_lw(self.conv3_lw(out[:,:,:,i,:]))

		for i in range(out1.shape[4]): #Lh
			out1[:,:,:,:,i] += self.bn3_lh(self.conv3_lh(out[:,:,:,:,i]))'''

		out1 = out1.permute(0,3,4,1,2)
		out1 = out1.contiguous().view(-1, 64, 120)

		out1 = self.bn2_l ( self.conv2_l(out1) )

		out1 = out1.contiguous().view(x.shape[0], 64, 120, 112, 112)

		out = self.maxpool(F.relu(out1, True))

		'''Third Conv'''
		out1 = torch.zeros([x.shape[0], 128, 60, 56, 56]).to(self.device)

		for i in range(out1.shape[2]): #hw
			out1[:,:,i,:,:] += F.relu (self.bn3_hw(self.conv3_hw(out[:,:,i,:,:])), True)

		'''for i in range(out1.shape[3]): #Lw
			out1[:,:,:,i,:] += self.bn3_lw(self.conv3_lw(out[:,:,:,i,:]))

		for i in range(out1.shape[4]): #Lh
			out1[:,:,:,:,i] += self.bn3_lh(self.conv3_lh(out[:,:,:,:,i]))'''

		out1 = out1.permute(0,3,4,1,2)
		out1 = out1.contiguous().view(-1, 128, 60)

		out1 = self.bn3_l ( self.conv3_l(out1) )

		out1 = out1.contiguous().view(x.shape[0], 128, 60, 56, 56)

		out = self.maxpool(F.relu(out1, True))

		'''Fourth Conv'''

		out1 = torch.zeros([x.shape[0], 128, 30, 28, 28]).to(self.device)

		for i in range(out1.shape[2]): #hw
			out1[:,:,i,:,:] += F.relu (self.bn4_hw(self.conv4_hw(out[:,:,i,:,:])), True)

		'''for i in range(out1.shape[3]): #Lw
			out1[:,:,:,i,:] += self.bn4_lw(self.conv4_lw(out[:,:,:,i,:]))

		for i in range(out1.shape[4]): #Lh
			out1[:,:,:,:,i] += self.bn4_lh(self.conv4_lh(out[:,:,:,:,i]))'''


		out1 = out1.permute(0,3,4,1,2)
		out1 = out1.contiguous().view(-1, 128, 30)

		out1 = self.bn4_l ( self.conv4_l(out1) )

		out1 = out1.contiguous().view(x.shape[0], 128, 30, 28, 28)

		out = self.maxpool(F.relu(out1, True))

		'''Classfn'''
		out = self.fc(out.view( out.shape[0], -1))

		out = self.probability(out)

		return out
