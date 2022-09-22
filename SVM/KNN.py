# comparing SVM with K Nearest Neighbours
import sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import pickle

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

X = cancer.data
Y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.2)

# print(x_train, y_train)

classes = ['malignant', 'benign']
k=3
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(x_train,y_train)
predictions = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, predictions)
print(acc)


'''best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
    k=3
    clf = KNeighborsClassifier(n_neighbors=k)

    clf.fit(x_train,y_train)

    predictions = clf.predict(x_test)

    acc = metrics.accuracy_score(y_test, predictions)

    if acc > best:
        best = acc
        with open("./models/KNN.pickle","wb") as f:
            pickle.dump(clf,f)

pickled_in = open("./models/KNN.pickle","rb")
loadKNN = pickle.load(pickled_in)

pred = loadKNN.predict(x_test)
accuracy = metrics.accuracy_score(y_test, pred)
print(accuracy)'''