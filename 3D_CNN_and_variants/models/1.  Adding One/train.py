import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loader import *
from model import *
torch.backends.cudnn.benchmark = True


num_epochs = 300
batch_size = 2
lr = 0.001

#data loading
train_data = VideoDataset(mode = "train")
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 8)
train_data_len = len(train_data)

val_data = VideoDataset(mode = "val")
val_loader = DataLoader(val_data, batch_size = batch_size, num_workers = 8)
val_data_len = len(val_data)

#model initialization and multi GPU
device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")

#resnet not pre-trained one
model = Model()

#using more than 1 GPU if available
#if (torch.cuda.device_count() > 1):
#   model = nn.DataParallel(model)
model = model.to(device)

#loss, optimizer and scheduler
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 5e-5)
scheduler = ReduceLROnPlateau(optimizer, cooldown = 5, min_lr = 0.0001, verbose = True)

best_acc = 0.0

print ("Training Started")
#training
for epoch in range(num_epochs):
	train_acc = 0.0
	epoch_loss = 0.0
	count = 0.0

	model.train()
	for i, (inputs,labels) in enumerate(train_loader):
		inputs = inputs.to(device) #change to device
		labels = labels.to(device)

		predictions = model(inputs) # predictions

		#now calculate the loss function
		loss = criterion(predictions.squeeze(), labels.float())

		#backprop here
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		epoch_loss += (loss.data * inputs.shape[0])
		predictions = predictions >= 0.5  # give abnormal label
		train_acc += (predictions.squeeze().float() == labels.float()).sum() #get the accuracy

		#print('Ep_Tr:{}/{},step:{}/{},top1:{},loss:{}'.format(epoch, num_epochs, i, train_data_len //batch_size,train_acc1, loss.data))
	epoch_loss = (epoch_loss / float(train_data_len))
	train_acc = train_acc.float()
	train_acc /= float(train_data_len)

	print("Ep_Tr: {}/{}, acc: {}, ls: {}".format(epoch, num_epochs, train_acc.item(), epoch_loss.data))


	#validation
	model.eval()
	epoch_loss = 0.0
	val_acc = 0.0

	for i, (inputs,labels) in enumerate(val_loader):
		inputs = inputs.to(device) #change to device
		labels = labels.to(device)

		with torch.no_grad():
			predictions = model(inputs)

		loss = criterion(predictions.squeeze(), labels.float())
		epoch_loss += (loss.data * inputs.shape[0])
		predictions = predictions >= 0.5  # give abnormal label

		val_acc += (predictions.squeeze().float() == labels.float()).sum()

	#print('Ep_vl: {}/{},step: {}/{},top1:{}'.format(epoch, num_epochs, i, test_data_len //batch_size,val_acc1))
	epoch_loss = (epoch_loss / float(val_data_len))
	val_acc = val_acc.float()
	val_acc /= float(val_data_len)

	print('Ep_vl: {}/{}, val acc: {}, ls: {}'.format(epoch, num_epochs, val_acc.data, epoch_loss.data))

	scheduler.step(epoch_loss.item()) #for the scheduler


	if (best_acc <= val_acc.data):
		best_acc = val_acc.data
		state = {'acc':best_acc,'epoch': epoch+1, 'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()}
		torch.save(state, "./best_3Dconv.pt")

	print('Epoch: {}/{}, best_acc: {}'.format(epoch, num_epochs, best_acc)) #print the epoch loss

	if (epoch % 10) == 0:
	  state = {'epoch':epoch+1, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict()}
	  torch.save(state, "./3Dconv.pt")
