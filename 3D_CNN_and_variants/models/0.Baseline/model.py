import torch
import torch.nn as nn
import torch.nn.functional as F 

import math
			 
class Model(nn.Module):
	
	def __init__(self):
		super(Model, self).__init__()

		self.conv1 = nn.Conv3d(3, 64, kernel_size = 3, stride = (2, 2, 2), padding = (1, 1, 1))
		self.bn1 = nn.BatchNorm3d(64)

		self.conv2 = nn.Conv3d(64, 64, kernel_size = 3, stride = (2, 2, 2), padding = (1, 1, 1))
		self.bn2 = nn.BatchNorm3d(64)

		self.conv3 = nn.Conv3d(64, 128, kernel_size = 3, stride = (2, 2, 2), padding = (1, 1, 1))
		self.bn3 = nn.BatchNorm3d(128)

		self.conv4 = nn.Conv3d(128, 128, kernel_size = 3, stride = (2, 2, 2), padding = (1, 1, 1))
		self.bn4 = nn.BatchNorm3d(128)

		self.fc = nn.Linear(128 * 15 * 14 * 14, 1)
		self.probability = nn.Sigmoid()

		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, x):
		
		out = self.bn1(F.relu(self.conv1(x)))

		out = self.bn2(F.relu(self.conv2(out)))
		
		out = self.bn3(F.relu(self.conv3(out)))

		out = self.bn4(F.relu(self.conv4(out)))

		out = self.fc(out.view( out.shape[0], -1))

		out = self.probability(out)

		return out