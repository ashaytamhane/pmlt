import numpy as np
import math

# Step 1: - Logistic regression is a parametric model. We will assume p(t|x) follows a bernoulli distribution (binary classification)
#         - p(t|x) = y_predicted^(y_actual)* (1-y_predicted)^(1-y_actual)
# Step 2: - Hypothesis function: y_predicted= sigmoid (W transpose * X) where sigmoid(a) = 1/(1+e^-a)
# Step 3: - Max likelihood: p(T|X)= product over all samples p(t|x)
#         - Log likelihood ln p(T|X)= product over all samples y_predicted^(y_actual)*(1-y_predicted)^(1-y_actual)
#                                   = product over all samples y_actual*ln y_predicted+ (1-y_actual)*ln (1-y_predicted)
# Step 4: - Optimise using gradient descent
#         - Minimise error E(W)= - ln p(T|X)
#         - Taking derivative: dE/dW= sum over all samples (y_predicted-y_actual)*x

class LogisticRegression:
	'''
		This class assumes X and Y to be input as N*D and N*1 dimensions respectively
	'''
	def __init__(self, X, Y):
		self.X=np.transpose(X) # convert into internal notation of D*N
		self.Y=np.transpose(Y) 
		self.num_samples, self.dim=X.shape 
		self.W=np.random.rand(self.dim,1)

	def forward(self,X):
		X=np.transpose(X) # get X in internal dim format
		Y=np.transpose(self.W) @ X
		Y_predicted= np.array([1/(1+math.exp(-i)) for i in Y[0]])
		return Y_predicted.reshape(1,len(Y_predicted))

	def loss(self,Y_predicted, Y_actual):
		return np.sum((Y_predicted-Y_actual)**2)/len(Y_predicted)

	# Since we are calculating gradient manually, defining a function to return gradient
	# The function returns D dimensional weights assuming X is D dimensional (N*D)	
	# Ys need to be N*1 format
	def error_gradient(self, Y_predicted, X):
		# the derivative of error function is just calculated "by hand" and coded below
		return (Y_predicted-self.Y) @ X
