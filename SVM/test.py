import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import pickle

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

X = cancer.data
Y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.2)

# print(x_train, y_train)

classes = ['malignant', 'benign']

# clf = svm.SVC()
clf = svm.SVC(kernel="linear", C=2)
# C= is soft margin
# svm.SVC(kernel="poly", degree=2)
clf.fit(x_train,y_train)

predictions = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, predictions)

print(acc)

'''best = 0
for _ in range(10): #dont go overboard with range cause it will take lot of time
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
    clf = svm.SVC(kernel="linear",C=2)
    clf.fit(x_train,y_train)
    predictions = clf.predict(x_test)
    acc = metrics.accuracy_score(y_test, predictions)

    if acc > best:
        best = acc
        with open("./models/SVM.pickle","wb") as f:
            pickle.dump(clf,f)

pickled_in = open("./models/SVM.pickle","rb")
loadSVM = pickle.load(pickled_in)

pred = loadSVM.predict(x_test)
accuracy = metrics.accuracy_score(y_test, pred)
print(accuracy)'''