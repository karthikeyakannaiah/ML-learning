import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv('student-mat.csv', sep=';')

data = data[["G1","G2","G3","studytime","failures","absences","goout"]]

predict="G3"

X = np.array(data.drop(labels=[predict],axis=1))
# features or attributes

Y = np.array(data[predict])
# labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.1)

''' best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)
    # print("current Accuracy ", acc)
    if acc > best:
        best = acc
        with open("./models/studentmodel.pickle","wb") as f:
            pickle.dump(linear, f)'''

pickle_in = open("./models/studentmodel.pickle","rb")
linear = pickle.load(pickle_in)
# print("best accuracy: ", best)
print("CoEf ", linear.coef_)
print("Intercept ", linear.intercept_)

predictions = linear.predict(x_test)
print("\npredictions: \n")
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
print(len(predictions)," ",len(x_test))

# predicting
# print("predicting [[7,4,2,0,5,2]]:\n",linear.predict([[7,4,2,0,5,2]]))

p = "G1" # "G2" "failures" etc.., check em out to see corelations
style.use("ggplot")
pyplot.scatter(data[p],data[predict])

pyplot.xlabel(p)
pyplot.ylabel("Final Grade (prediction)")
pyplot.show()
