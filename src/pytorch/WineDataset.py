from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from sklearn import preprocessing

class WineDataset(Dataset):
	'''
	    Class for custom data set. Pls tweak the __init__ function for custom data processing
	'''

	def __init__(self):
		super().__init__()
		# since the data is all numerical, can use loadtxt function
		data=np.loadtxt("/Users/ashay.tamhane/pmlt/data/wine.csv",delimiter=",",skiprows=1)

		# split features and label. Label feature is 0
		X=data[:,1:]
		Y=[0 if i<3 else 1 for i in data[:,0]]
		self.Y=torch.tensor(Y,dtype=torch.float32)

		# normalise the features. Store the scaler
		scaler=preprocessing.StandardScaler()
		self.scaler=scaler.fit(X)
		self.X=torch.tensor(scaler.transform(X),dtype=torch.float32)
		self.num_samples,self.num_features=X.shape

	def __getitem__(self,index):
		return self.X[index],self.Y[index]

	def __len__(self):
		return len(self.X)