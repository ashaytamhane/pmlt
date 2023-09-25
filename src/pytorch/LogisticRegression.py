import torch.nn as nn

class LogisticRegression(nn.Module):
	'''
		Define logistic regression model

	'''

	def __init__(self,input_size,output_size):
		super().__init__()
		self.lin=nn.Linear(input_size,output_size)
		self.sig=nn.Sigmoid()

	def forward(self,X):
		return self.sig(self.lin(X))


class LogisticRegressionScratch():
	'''
	    To be implemented from basic torch operations
	'''
	pass

