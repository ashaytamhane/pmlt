import torch
import torch.nn as nn
import numpy as np
from WineDataset import WineDataset
from torch.utils.data import DataLoader, Dataset
from LogisticRegression import LogisticRegression


# load dataset
full_dataset=WineDataset()
full_dataset_len=len(full_dataset)
print(f'Full data shape: {full_dataset_len}')

# split into train and test
train_data_size=int(0.8*full_dataset_len)
test_data_size=int(full_dataset_len-train_data_size)
train_dataset, test_dataset=torch.utils.data.random_split(full_dataset,[train_data_size,test_data_size],generator=torch.Generator().manual_seed(0))

print(f'Train data len: {len(train_dataset)}, Test data len: {len(test_dataset)}')


# training 
num_iter=50
learning_rate=0.1
batch_size=10
dl=DataLoader(train_dataset,batch_size=10,shuffle=True)
model=LogisticRegression(full_dataset.num_features,1)
l=nn.BCELoss()
optimiser=torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(0,num_iter):

	# iterate over the batches
	for i, (X,Y) in enumerate(dl):
		# training for each batch
		
		#forward pass
		Y_predicted=model(X)

		# calculate loss
		loss=l(Y_predicted,Y.reshape(-1,1))

		# gradients
		loss.backward()

		# optimise
		optimiser.step()
		optimiser.zero_grad()


	if(epoch%5==0):	
		print(f'Epoch: {epoch}, Loss: {loss:.02f}')

# test loop
model.eval()

tl=DataLoader(test_dataset,batch_size=len(test_dataset))
correct=0
with torch.no_grad():

	# iterate through test batches
	for i, (X,Y) in enumerate(tl):
		# standardise the features
		X_scaled=torch.tensor(full_dataset.scaler.transform(X),dtype=torch.float32)
		Y_test_predicted=model(X_scaled)

		# accuracy of entire batch
		acc= (Y_test_predicted.round()==Y)
		print(f'Accuracy: {acc.float().mean()*100}')
