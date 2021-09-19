import matplotlib.pyplot as plt

train_loss = []
val_loss = []

train_acc = []
val_acc = []

best_acc = []

f = open("loss.txt")

for line in f:
	if ("Ep_Tr" in line):
		tmp = line.rstrip().split(",")

		acc = float(tmp[1].split(":")[-1])

		loss = float(tmp[-1].split(":")[-1])

		train_acc.append(acc)
		train_loss.append(loss)

	elif ("Ep_vl" in line):
		tmp = line.rstrip().split(",")

		acc = float(tmp[1].split(":")[-1])

		loss = float(tmp[-1].split(":")[-1])

		val_acc.append(acc )
		val_loss.append(loss) 

	else:
		continue

plt.xlabel("Number of Epochs")
plt.ylabel("Train and Val Losses")
plt.title("Joint Losses Plot")
plt.plot(train_loss,  label = "train")
plt.plot(val_loss,  label = "val")
plt.legend()
plt.savefig('losses.png')

plt.clf()

plt.xlabel("Number of Epochs")
plt.ylabel("Train and Val Accs")
plt.title("Joint Accs Plot")
plt.plot(train_acc,  label = "train")
plt.plot(val_acc,  label = "val")
plt.legend()
plt.savefig('accs.png')

