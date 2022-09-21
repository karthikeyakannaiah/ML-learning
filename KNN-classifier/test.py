import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style

data = pd.read_csv('car.data')
# print(data.head(10))
le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
# print(buying)
predict = 'class'

k = 3


X = list(zip(buying,maint,door,persons,lug_boot,safety))
Y = list(cls)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.05)
best=0
'''for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.05)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train,y_train)
    acc = model.score(x_test,y_test)
    if acc > best:
        best = acc
        with open("./models/carEval.model","wb") as f:
            pickle.dump(model,f)'''

pickled = open("./models/carEval.model","rb")
modelLoaded = pickle.load(pickled)
accuracy = modelLoaded.score(x_test,y_test)
print(accuracy)


predictions = modelLoaded.predict(x_test)
# predictions = le.inverse_transform(list(predictions))
# Ytest = le.inverse_transform(list(y_test))
names = ["unacc","acc","good","vgood"]
for i in range(len(predictions)):
    print(names[predictions[i]]," ",names[y_test[i]])
    print(modelLoaded.kneighbors([x_test[i]], k, True))
    
# style.use('ggplot')
# pyplot.scatter(data['buying'],data[predict])
# pyplot.xlabel("buying")
# pyplot.ylabel(predict)
# pyplot.show()