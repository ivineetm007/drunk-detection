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

		#self.conv1 = nn.Conv3d(3, 64, kernel_size = 3, stride = (2, 2, 2), padding = (1, 1, 1))
		#self.bn1 = nn.BatchNorm3d(64)

		#self.conv2 = nn.Conv3d(64, 64, kernel_size = 3, stride = (2, 2, 2), padding = (1, 1, 1))
		#self.bn2 = nn.BatchNorm3d(64)

		'''First Conv'''
		'''self.conv1_h = nn.Conv1d(3, 64, kernel_size = 3, padding = 1)
		self.bn1_h = nn.BatchNorm1d(64)

		self.conv1_w = nn.Conv1d(3, 64, kernel_size = 3, padding = 1)
		self.bn1_w = nn.BatchNorm1d(64)

		self.conv1_l = nn.Conv1d(3, 64, kernel_size = 3, padding = 1)
		self.bn1_l = nn.BatchNorm1d(64)'''

		self.conv1 = nn.Conv3d(3, 64, kernel_size = 3, stride = (2, 2, 2), padding = (1, 1, 1))
		self.bn1 = nn.BatchNorm3d(64)

		self.conv2 = nn.Conv3d(64, 64, kernel_size = 3, stride = (2, 2, 2), padding = (1, 1, 1))
		self.bn2 = nn.BatchNorm3d(64)


		'''Second Conv'''
		'''self.conv2_h = nn.Conv1d(64, 64, kernel_size = 3, padding = 1)
		self.bn2_h = nn.BatchNorm1d(64)

		self.conv2_w = nn.Conv1d(64, 64, kernel_size = 3, padding = 1)
		self.bn2_w = nn.BatchNorm1d(64)

		self.conv2_l = nn.Conv1d(64, 64, kernel_size = 3, padding = 1)
		self.bn2_l = nn.BatchNorm1d(64)'''

		'''Third one'''
		self.conv3_h = nn.Conv1d(64, 128, kernel_size = 3, padding = 1)
		self.bn3_h = nn.BatchNorm1d(128)

		self.conv3_w = nn.Conv1d(64, 128, kernel_size = 3, padding = 1)
		self.bn3_w = nn.BatchNorm1d(128)

		self.conv3_l = nn.Conv1d(64, 128, kernel_size = 3, padding = 1)
		self.bn3_l = nn.BatchNorm1d(128)

		#self.conv3_lh = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
		#self.bn3_lh = nn.BatchNorm2d(128)

		#self.conv3_lw = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
		#self.bn3_lw = nn.BatchNorm2d(128)


		'''Fourth One'''
		self.conv4_h = nn.Conv1d(128, 128, kernel_size = 3, padding = 1)
		self.bn4_h = nn.BatchNorm1d(128)

		self.conv4_w = nn.Conv1d(128, 128, kernel_size = 3, padding = 1)
		self.bn4_w = nn.BatchNorm1d(128)

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
		'''out1 = x
		out = torch.zeros([x.shape[0], 64, 240, 224, 224]).to(self.device)

		#h conv
		out1 = x.permute(0,2,3,1,4)
		out1 = out1.contiguous().view(-1, 3, 224)

		out1 = self.bn1_h( self.conv1_h( out1) )
		out1 = F.relu(out1, True)

		out1 = out1.contiguous().view(x.shape[0], 64, 240, 224, 224)
		out += out1

		#w conv
		out1 = x.permute(0,2,4,1,3)
		out1 = out1.contiguous().view(-1, 3, 224)

		out1 = self.bn1_w( self.conv1_w( out1) )
		out1 = F.relu(out1, True)

		out1 = out1.contiguous().view(x.shape[0], 64, 240, 224, 224)
		out += out1

		#l conv
		out1 = x.permute(0,3,4,1,2)
		out1 = out1.contiguous().view(-1, 3, 224)

		out1 = self.bn1_l( self.conv1_l( out1) )
		out1 = F.relu(out1, True)

		out1 = out1.contiguous().view(x.shape[0], 64, 240, 224, 224)
		out += out1

		out = self.maxpool(F.relu(out, True))'''

		out = self.bn1(F.relu(self.conv1(x)))
		out = self.bn2(F.relu(self.conv2(out)))

		'''Second Conv'''
		'''out1 = out
		out = torch.zeros([x.shape[0], 64, 120, 112, 112]).to(self.device)

		#h conv
		out1 = out.permute(0,2,3,1,4)
		out1 = out1.contiguous().view(-1, 64, 112)

		out1 = self.bn2_h( self.conv2_h( out1) )
		out1 = F.relu(out1, True)

		out1 = out1.contiguous().view(x.shape[0], 64, 120, 112, 112)
		out += out1

		#w conv
		out1 = out.permute(0,2,4,1,3)
		out1 = out1.contiguous().view(-1, 64, 112)

		out1 = self.bn2_w( self.conv2_w( out1) )
		out1 = F.relu(out1, True)

		out1 = out1.contiguous().view(x.shape[0], 64, 120, 112, 112)
		out += out1

		#l conv
		out1 = out.permute(0,3,4,1,2)
		out1 = out1.contiguous().view(-1, 64, 112)

		out1 = self.bn2_l( self.conv2_l( out1) )
		out1 = F.relu(out1, True)

		out1 = out1.contiguous().view(x.shape[0], 64, 120, 112, 112)
		out += out1

		out = self.maxpool(F.relu(out, True))'''


		'''Third Conv'''
		out1 = out
		out2 = torch.zeros([x.shape[0], 128, 60, 56, 56]).to(self.device)

		#h conv
		out1 = out.permute(0,2,3,1,4)
		out1 = out1.contiguous().view(-1, 64, 56)

		out1 = self.bn3_h( self.conv3_h( out1) )
		out1 = F.relu(out1, True)

		out1 = out1.contiguous().view(x.shape[0], 128, 60, 56, 56)
		out2 += out1

		#w conv
		out1 = out.permute(0,2,4,1,3)
		out1 = out1.contiguous().view(-1, 64, 56)

		out1 = self.bn3_w( self.conv3_w( out1) )
		out1 = F.relu(out1, True)

		out1 = out1.contiguous().view(x.shape[0], 128, 60, 56, 56)
		out2 += out1

		#l conv
		out1 = out.permute(0,3,4,1,2)
		out1 = out1.contiguous().view(-1, 64, 56)

		out1 = self.bn3_l( self.conv3_l( out1) )
		out1 = F.relu(out1, True)

		out1 = out1.contiguous().view(x.shape[0], 128, 60, 56, 56)
		out2 += out1

		out = self.maxpool(F.relu(out2, True))

		'''Fourth Conv'''
		out1 = out
		out2 = torch.zeros([x.shape[0], 128, 30, 28, 28]).to(self.device)

		#h conv
		out1 = out.permute(0,2,3,1,4)
		out1 = out1.contiguous().view(-1, 128, 28)

		out1 = self.bn4_h( self.conv4_h( out1) )
		out1 = F.relu(out1, True)

		out1 = out1.contiguous().view(x.shape[0], 128, 30, 28, 28)
		out2 += out1

		#w conv
		out1 = out.permute(0,2,4,1,3)
		out1 = out1.contiguous().view(-1, 128, 28)

		out1 = self.bn4_w( self.conv4_w( out1) )
		out1 = F.relu(out1, True)

		out1 = out1.contiguous().view(x.shape[0], 128, 30, 28, 28)
		out2 += out1

		#l conv
		out1 = out.permute(0,3,4,1,2)
		out1 = out1.contiguous().view(-1, 128, 28)

		out1 = self.bn4_l( self.conv4_l( out1) )
		out1 = F.relu(out1, True)

		out1 = out1.contiguous().view(x.shape[0], 128, 30, 28, 28)
		out2 += out1

		out = self.maxpool(F.relu(out2, True))

		'''Classfn'''
		out = self.fc(out.view( out.shape[0], -1))

		out = self.probability(out)

		return out
