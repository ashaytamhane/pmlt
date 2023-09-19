from kNN import kNN
from myutils import Dataloader
import numpy as np

# load data 
dl= Dataloader("/Users/ashay.tamhane/Documents/iris.csv",label_column_names=['variety'])
dl.stats()

# call knn
knn=kNN(dl.X_train_numeric,dl.Y_train,5)
predictions=knn.predict(dl.X_test_numeric)

# print accuracy
acc=np.sum(predictions==dl.Y_test)/ len(dl.Y_test)
print(f'Accuracy: {acc*100:.02f}') 