# K-Nearest Neighbor Classifier with Scikit Learn
K-NN is a very simple machine learning algorithm which can classify a point based on other nearest points. Let's take an example, if you see the below image.

We have set k = 3, this means that we will classify a point based on the nearest three points, in this case two of the three points are orange points therefore the unknown point (blue point) will be classified as an orange point.

![enter image description here](https://media.licdn.com/dms/image/C5112AQFhHCHjjYf7xQ/article-inline_image-shrink_1500_2232/0?e=1540425600&v=beta&t=RqoTohXHjlf-QVkJqU8hhygtXmv4xGcPWnZyHv2hVKE)

OK, let's do a real classification task, in this example I am going to classify the most popular iris flower data set. In this data set there are samples of three different types of Iris flower. Those are **Versicolor, Verginica, Setosa** and I am going to build a model that is capable of classifying a new flower into one of these categories.

![enter image description here](https://media.licdn.com/dms/image/C5112AQFl-85rxGLBWQ/article-inline_image-shrink_1000_1488/0?e=1540425600&v=beta&t=kXlzdsy66pVjkDgY-hyy4HHyudgJf-DMcmKzdnpashs)

Let's code,

**Step 1 - Import the necessary libraries and load the data (I am going to make use of Scikit Learn's Iris Data set)**

    from sklearn import datasets
    import numpy as np
    
    iris_dataset = datasets.load_iris()

Let's understand the data set, this data set consist of 150 records (Every Iris type has 50 records each.) which have information like flower's sepal length, sepal width, petal length, petal width and the label/class.

If you look at the **"iris_dataset"** variable above, it is a python dictionary with 5 key value pairs.
1. DESCR - Description of the dataset

2. data - Feature values for all 150 records (sepal length, sepal width, petal length, petal width)

3. feature_name - Information about the features

4. target - classes of all the 150 records

5. target_name - names of the classes/ flowers.

But among these I am only interested in data and target. In other words features and label.

![](https://media.licdn.com/dms/image/C5112AQGsgPwFNN7UVQ/article-inline_image-shrink_1000_1488/0?e=1540425600&v=beta&t=axWfjXXIoMT2UPWcTXvLh2swn1kyHvtXLiE_E5VmlME)

![enter image description here](https://media.licdn.com/dms/image/C5112AQGsgPwFNN7UVQ/article-inline_image-shrink_1000_1488/0?e=1540425600&v=beta&t=axWfjXXIoMT2UPWcTXvLh2swn1kyHvtXLiE_E5VmlME)

**Step 2 - Prepare the data set**

    # get the features and labels
    features = np.array(iris_dataset.data, dtype="float64")
    labels = np.array(iris_dataset.target, dtype="float64")
    
    # combine featutre with labels to shuffle the ordered data
    data = np.column_stack((features, labels))
    np.random.shuffle(data)
    
    # preparing dataset and label dataset
    X = data[:, :-1]
    y = data[:, -1]

Let's break the above code.

First step is so obvious, getting the features and labels into variables. ( There is no array type of data structure in python so we use numpy arrays )

Before discussing about the second part let's discuss why we need this step, In our data set we have the labels and features separately, which is good in a way because at the end we will separate the features and labels to train the model.

but the real problem is, as you can see in the above image the records in the data set are grouped, which means the first 50 records belongs to class 0 (Setosa), the next 50 records are class 1 (Versicolor), and the final 50 belongs to class 2 (Virginica). This will create a bias model, which is not what we want. So we need to shuffle the data set first.

In ordered to shuffle the data set, first we will concatenate the separated features and labels into one array, and then we will shuffle the array. That's it, we have finished preparing our data set.

![enter image description here](https://media.licdn.com/dms/image/C5112AQFaG63nhLu9eA/article-inline_image-shrink_1500_2232/0?e=1540425600&v=beta&t=ltRcto-5OrE-_yVcuN9-yqr5CjEh03RncyIuWDNoOdI)

In the final step, we have split the features and labels again and stored it into X and y variables.

**Step 3 - Let's train and test our model.**

    # training and testing set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    
    # train the model
    from sklearn import neighbors
    clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    
    # accuracy testing 
    accuracy = clf.score(X_test, y_test)

1. First we will separate our data set into two parts, one will be used to train the model and the other one will be used to test the model, the training and testing ratio can be anything, but it's preferred to use 80% of the data for training and 20% to test the accuracy of the model. This can be done very easily with train_test_split of sklearn.model_selection.

2. After that we create the KNeighborsClassifier model and fit (train) the training data to the model. An important thing to note here is, I have passed an argument to the Classifier, what is that? That is nothing but the K value I talked about earlier, In this case I have set 5 for that parameter (default value for this parameter is also five, but for the sack of explanation I specified it), that means the model will classify a point based on the nearest 5 points. We have to be careful when choosing the values for K, so let's discuss why I choose 5 for this.

-   **Our data set have 3 different classes so we cannot choose 3, because there is a possibility that all the classes get 1 nearest point each, not only for 3 but any number that is divisible by three will have the same issue.**

-   **What about k = 4, not only four but using any even number is not recommended here, because there is a very high possibility for some classes to have same number of nearest points.**

-   **We can give any odd numbers that are not divisible by 3, since our data set is small I choose the first number which fit our criteria that is 5.**

3. Finally, either we can predict the class for new feature set or we can find how accurate our model is using the test set. In this case I am only interested in finding the accuracy of the model. So I used score function of the Classifier Object and passed the test set.

That's it.