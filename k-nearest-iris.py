from sklearn import datasets
import numpy as np
#import pandas as pd

iris_dataset = datasets.load_iris()

features = np.array(iris_dataset.data, dtype="float64")
labels = np.array(iris_dataset.target, dtype="float64")


#combine featutre with labels to shuffle the ordered data
data = np.column_stack((features, labels))
np.random.shuffle(data)

X = data[:, :-1]
y = data[:, -1]

# training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# train the model
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

# accuracy testing 
accuracy = clf.score(X_test, y_test)
clf_pred = clf.predict(X_test)

for i in range(len(y_test)):
    print(clf_pred[i], y_test[i], '\n')

print(accuracy)

