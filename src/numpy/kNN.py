from myutils import Dataloader
import numpy as np
from collections import Counter

# kNN is a non-parametric approach as it does not make any assumptions on the data
# In kNN, we simply fetch k nearest neighbours in terms of euclidean distance and take their majority votes (labels) for prediction

class kNN:
	def __init__(self,X_train,Y_train,k):
		self.X_train=X_train
		self.Y_train=Y_train
		self.k=k

	def dist(self, x1, x2):
		return np.sqrt(np.sum((x1-x2)**2))

	def predict(self,X_test):
		Y_predicted=[self._predict(x_test) for x_test in X_test]
		return np.array(Y_predicted).reshape(-1,1)

	def _predict(self, x_test):
		# get k nearest neighbours of sample to be predicted
		distances=[self.dist(x_test,x_train) for x_train in self.X_train]
		closest_indices=np.argsort(distances)[:self.k]
		labels_k_neighbours=[self.Y_train[i][0] for i in closest_indices]

		# get the most common class label
		most_common= Counter(labels_k_neighbours).most_common(1)
		name,ind=most_common[0]
		return name








