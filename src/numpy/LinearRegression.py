import numpy as np
from myutils import Dataloader

# Theory and approach
# Step 1: - Linear Regression is a parametric model. We will assume that the data is drawn from a normal distribution
#         - p(t|x)= gaussian with mean y(w,x) and std deviation beta

# Step 2: - Hypothesis function: p(t|x)= W transpose*X where W is the column weight matrix and each col in X is a D dim sample
#         - We could also allow a transformation phi on x for non-linearity in x. So y(W,X) = Y_predicted= W transpose* phi (X)
#         - However, in this library its assumed all such transformations are applied in X itself

# Step 3: - Max likelihood approach: p(T|X)= product over all samples p(t|x) = product over gaussian with mean (W transpose * phi (X))
#         - Taking log likelihood, we will get the negative of error term 1/2*sum (y_predicted-y_actual)**2
#         - If we take derivative of the error term, we get dE/dW= sum (y_predicted-y_actual)* phi(X)

# Step 4: - Since the error function is convex, setting its derivative to 0 can give us min value of error
#           While the analytical solution is W_min= inverse (phi transpose phi)* phi transpose (pseudo penrose inverse),
#           we can simply implement gradient descent to minimise the error by using the derivative to avoid heavy operations

class LinearRegression:
	'''
		This class assumes that the data matrices X and Y are of shape N*D and N*1 where D is the number of dimensions/features
		and N is the number of samples
	'''

	def __init__(self, X,Y):
		self.X=np.transpose(X)
		self.Y=np.transpose(Y) 
		self.num_samples, self.dim=X.shape 
		# weight matrix will be a column matrix of dimension D*1
		self.W=np.random.rand(self.dim,1)


	def loss(self,Y_predicted, Y_actual):
		return np.sum((Y_predicted-Y_actual)**2)/len(Y_predicted)

	def forward(self, X):
		X=np.transpose(X) # to convert into internal D*N format
		return np.transpose(self.W) @ X

	# Since we are calculating gradient manually, defining a function to return gradient
	# The function returns D dimensional weights assuming X is D dimensional (D*N)	
	# Ys need to be 1*N format
	def error_gradient(self, Y_predicted, X):
		# the derivative of error function is just calculated "by hand" and coded below
		return (Y_predicted-self.Y) @ X



