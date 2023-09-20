import numpy as np
from myutils import Dataloader
from LinearRegression import LinearRegression

#load data
dl=Dataloader(filename="/Users/ashay.tamhane/Documents/regression_data.csv",label_column_names=['f2'])
dl.stats()

# initialise model
model=LinearRegression(dl.X_train_numeric, dl.Y_train)
# training loop
num_iter=10
learning_rate=0.01
for epoch in range(0,num_iter):

	# forward pass
	Y_predicted= model.forward(dl.X_train_numeric)
	
	# get error
	loss=model.loss(Y_predicted,dl.Y_train)
	print(f'Loss: {loss:0.2f}')

	# get gradient
	error_gradients_weights= model.error_gradient(Y_predicted,dl.X_train_numeric)
	
	# optimise with gradient descent
	model.W=model.W-learning_rate*error_gradients_weights

# predict for new data
Y_predicted=model.forward(dl.X_test_numeric)
print(f'X_test: {dl.X_test_numeric} Y_predicted: {Y_predicted}')
#accuracy
loss=model.loss(Y_predicted,dl.Y_test)
print(loss)