import torch
import torch.nn as nn
import numpy as np
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
from WineDataset import WineDataset
from MLP import MLP

# load data

full_dataset= WineDataset()
train_size= int(0.8*len(full_dataset))
test_size=len(full_dataset)-train_size
train_dataset, test_dataset= torch.utils.data.random_split(full_dataset, [train_size,test_size],generator=torch.Generator().manual_seed(0))

print(f'Train dataset: {len(train_dataset)}, Test dataset: {len(test_dataset)}')

batch_size=50
train_dataloader= DataLoader(train_dataset, batch_size=batch_size,shuffle=True)

# training loop
num_iter=100
num_features=full_dataset.num_features
num_hidden_layers=10
model=MLP(num_features,num_hidden_layers,1)
l=nn.BCELoss()
learning_rate=0.1
optimiser=torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(0,num_iter):
	for i, (X,Y) in enumerate(train_dataloader):
		# training of the batch

		# forward pass
		Y_predicted=model(X)

		# loss and derivatives
		loss=l(Y_predicted.squeeze(),Y)
		loss.backward()

		# optimiser
		optimiser.step()
		optimiser.zero_grad()

	if(epoch%10==0):
		print(f'Loss: {loss:.02f}')

# test loop
model.eval()
test_batch_size=len(test_dataset)
test_dataloader=DataLoader(test_dataset,batch_size=test_batch_size)

with torch.no_grad():
	for i, (X,Y) in enumerate(test_dataloader):
		scaled_X=torch.tensor(full_dataset.scaler.transform(X), dtype=torch.float32)
		Y_predicted_test=model(scaled_X)
		accuracy=(Y_predicted_test.round()==Y).float().mean()*100

	print(f'Accuracy: {accuracy:.02f}')

