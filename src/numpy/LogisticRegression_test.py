import numpy as np 
from myutils import Dataloader 
from LogisticRegression import LogisticRegression


# load data
dl=Dataloader(filename="/Users/ashay.tamhane/Documents/wine.csv",label_column_names=['Wine'],norm=True)
dl.stats()

#Since the data contains more than two labels but our class only supports binary
#we will convert all labels>1 as 1 and labels<1 as 0 for demo purposes
Y_train=[0 if i<2 else 1 for i in dl.Y_train]
Y_test= [0 if i<2 else 1 for i in dl.Y_test]

# initialise model
model=LogisticRegression(dl.X_train_numeric,dl.Y_train)

# training loop
num_iter=50
learning_rate=0.01
for epoch in range(0,num_iter):
	# forward pass
	Y_predicted=model.forward(dl.X_train_numeric)

	# compute loss
	loss=model.loss(Y_predicted,Y_train)
	print(f'Epoch: {epoch}, Loss: {loss:0.2f}')

	# optimise with gradient descent
	error_gradients_weights= model.error_gradient(Y_predicted,dl.X_train_numeric)
	model.W= model.W - learning_rate* error_gradients_weights

# test
Y_predicted=model.forward(dl.X_test_numeric)
loss=model.loss(Y_predicted, Y_test)
print(f'Test loss: {loss:.02f}')
