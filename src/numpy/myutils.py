import numpy as np

class Dataloader:

	'''
		This class helps load and pre-process data. This class mandates that the data file should have a header. 
	'''

	def __init__(self,filename,label_column_names, categorical_column_names=None,missing_values=' ',filling_values=None, delim=",", train_test_ratio=0.8,seed=0):
		
		# load data from the file
		self.data=np.genfromtxt(filename,skip_header=0,names=True,
			                   missing_values=missing_values, filling_values=filling_values, 
			                   delimiter=delim, encoding=None, dtype=None)

		self.num_samples,=self.data.shape

		# create a list of all column names that are not to be treated as numerical input features
		str_names=label_column_names.copy()
		if(categorical_column_names is not None):
			str_names.extend(categorical_column_names)

		# extract numeric, categorical and label columns
		numeric_feature_names= [i for i in self.data.dtype.names if i not in str_names]
		numeric_features=self.data[numeric_feature_names]
		categorical_features=self.data[categorical_column_names]
		labels=self.data[label_column_names]

		#convert tuples from ndarray to a 2D array by converting each tuple into a list
		self.numeric_features=np.array([list(numeric_features[i]) for i in range(self.num_samples)])
		if(categorical_column_names is not None):
			self.categorical_features=np.array([list(categorical_features[i]) for i in range(self.num_samples)])
		else:
			self.categorical_features=None
		self.labels=np.array([list(labels[i]) for i in range(self.num_samples)])

		# split into train/test
		idx=np.arange(0,self.num_samples)
		np.random.seed(seed)
		np.random.shuffle(idx)
		split_ind=int(np.round(self.num_samples*train_test_ratio))
		training_idx=idx[0:split_ind]
		test_idx=idx[split_ind:]
		
		self.X_train_numeric=np.array([self.numeric_features[i] for i in training_idx])
		self.X_test_numeric=np.array([self.numeric_features[i] for i in test_idx])
		self.Y_train=np.array([self.labels[i] for i in training_idx])
		self.Y_test=np.array([self.labels[i] for i in test_idx])

		if(categorical_column_names is not None):
			self.X_train_cat=np.array([self.categorical_features[i] for i in training_idx])
			self.X_test_cat=np.array([self.categorical_features[i] for i in test_idx])
		else:
			self.X_train_cat=None
			self.X_test_cat=None

	def stats(self):
		print(f'Number of samples: {self.num_samples}')
		print(f'Numeric input data shape: {self.numeric_features.shape}')
		print(f'Target data shape: {self.labels.shape}')
		print(f'Training input data shape: {self.X_train_numeric.shape}')
		print(f'Training target data shape: {self.Y_train.shape}')
		print(f'Test input data shape: {self.X_test_numeric.shape}')
		print(f'Test target data shape: {self.Y_test.shape}')


