# # 辅助Tpit图像分割，二分类


import os 
import sys
import torch
import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np

batch_size = 8
PRINT_PER_STEPS = 10

class Net(nn.Module):
	def __init__(self):
		super().__init__()  
		self.pool = nn.MaxPool2d(2, 2)
		self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
		self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
		self.conv3 = nn.Conv2d(16, 6, 3, padding=1)
		self.conv4 = nn.Conv2d(6, 2, 3, padding=1)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))# F.relu替换成torch.sigmoid
		x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = torch.mean(x, dim=(2, 3)) 
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
		return x

def init_weights(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') 
		
		
		m.bias.data.fill_(0)


MAX_SIZE = 100

def transform(image):
	totensor = transforms.ToTensor()
	image = totensor(image)
	result = np.ones((3,MAX_SIZE,MAX_SIZE), dtype=np.float32)
	size1 = image.size(1)
	size2 = image.size(2)
	pad1 = (MAX_SIZE-size1)//2
	pad2 = (MAX_SIZE-size2)//2
	result[:,pad1:pad1+size1,pad2:pad2+size2] = image
	
	return result


def main():
	cd_path = sys.argv[0]  # app.cd_path,'\\CNN\\CNN_multifunction_Matlab.py
	
	cd_path = cd_path.split("\\CNN\\")[0]
	print("cd_path:",cd_path)
	weights_file = "model.weights" 
	if sys.argv[1] == "train":   
		train = True
		if len(sys.argv) == 3:
			initial_seed = int(sys.argv[2])
		else:
			initial_seed = None
	else:
		assert sys.argv[1] == "test"
		train = False

	net = Net()

	if train:
		if initial_seed == None:
			print("Random initial seed:", torch.random.initial_seed())
		else:
			torch.random.manual_seed(initial_seed) # Copy the best initial_seed as 873069710100
			
			
			print("Random initial seed:", initial_seed)
		net.apply(init_weights)
		net.train()
		
		# print(os.path.abspath("."))
		
		trainset = torchvision.datasets.ImageFolder(cd_path+'\\CNN\CNN_train', transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)#, num_workers=2)

		#criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1.5,0.5]))
		criterion = nn.CrossEntropyLoss()
		# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
		# optimizer = optim.Adam(net.parameters())
		optimizer = optim.Adam(net.parameters()) # Adadelta,Adagrad,AdamW,SparseAdam?,Adamax,ASGD,LBFGS?,NAdam?,RAdam,RMSprop,Rprop
		
		
		try:
			for epoch in range(3):  # loop over the dataset multiple times
				running_loss = 0.0
				for i, data in enumerate(trainloader, 0):
					
					# get the inputs
					inputs, labels = data

					
					optimizer.zero_grad()

					# forward + backward + optimize
					outputs = net(inputs)
					loss = criterion(outputs, labels)
					loss.backward()
					optimizer.step()

					
					running_loss += loss.item()
					if i % PRINT_PER_STEPS == PRINT_PER_STEPS-1:    # print every 2000 mini-batches
						print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / PRINT_PER_STEPS))
						
						running_loss = 0.0
		except KeyboardInterrupt:
			pass
		torch.save(net.state_dict(), weights_file)
		
		torch.save(net.state_dict(), cd_path+"\\CNN\\examples\\cancer_classification.model")
	else:
		net.load_state_dict(torch.load(cd_path+"\\CNN\\examples\\1\\cancer_classification.model"))
		# Adadelta,Adagrad,AdamW,SparseAdam?,Adamax,ASGD,LBFGS?,NAdam?,RAdam,RMSprop,Rprop
		
		net.eval()

		testset = torchvision.datasets.ImageFolder(cd_path+'\\CNN\CNN_Matlab_test', transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle = False)#, num_workers=2)

		classes = ('normal_cell', 'cancer_cell')

		all_labels = []
		all_predictions = []
		correct_overall = 0
		correct_per_class = [0, 0]
		total_overall = 0
		total_per_class = [0, 0]
		with torch.no_grad():
			for data in testloader:
				images, labels = data
				outputs = net(images)
				outputs = F.softmax(outputs, dim=1)
				
				outputs = outputs.numpy()
				labels = labels.numpy()
				predicted = np.argmax(outputs, axis=1)
				
				all_labels.append(labels)
				
				all_predictions.append(predicted) 
				correct = predicted == labels
				total_overall += labels.shape[0]
				
				correct_overall += correct.sum()
				for n in [0, 1]:
					this_class = labels == n
					total_per_class[n] += this_class.sum()
					correct_per_class[n] += correct[this_class].sum()
					
					
		
		np.save(cd_path+"\\CNN\\CNN_Matlab_test\labels", np.concatenate(all_labels))  
		np.savetxt(cd_path+"\\CNN\\CNN_Matlab_test\\test_results.txt", np.concatenate(all_predictions))
		print('Accuracy on the test images: %.4f %%' % (100 * correct_overall / total_overall))
		
		for i in [0, 1]:
			print('Accuracy of %s : %.4f %%' % (classes[i], 100 * correct_per_class[i] / total_per_class[i])) 
			


if __name__ == "__main__": 
	 
	main()

### "D:\\Python64\\python.exe" -m pdb D:\\Github\\CRCS\\CNN\\CNN_multifunction_Matlab.py test